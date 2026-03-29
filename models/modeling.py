# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models.configs as configs
from models.modules import Block as IELT_Block
from models.IEPV import MultiHeadVoting

from .modeling_resnet import ResNetV2
from .supcon_losses import SupConLoss
import torch.nn.functional as F


logger = logging.getLogger(__name__)




def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            n_patches = (img_size[0] // 16) * (img_size[1] // 16)
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers,
                                         width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.mask_patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, mask):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        mask = mask * x

        if self.hybrid:
            x = self.hybrid_model(x)
        x = self.patch_embeddings(x)
        x = x.flatten(2)
        x = x.transpose(-1, -2)
        x = torch.cat((cls_tokens, x), dim=1)


        mask = self.mask_patch_embeddings(mask)
        mask = mask.flatten(2)
        mask = mask.transpose(-1, -2)
        mask = torch.cat((cls_tokens, mask), dim=1)

        embeddings = x + mask + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Sequence_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Sequence_Embeddings, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 300+2352*0+1, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings

class Time_Sequence_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Time_Sequence_Embeddings, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1+625, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Space_Sequence_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config):
        super(Space_Sequence_Embeddings, self).__init__()
        self.position_embeddings = nn.Parameter(torch.zeros(1, 1+12+12, config.hidden_size))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        B = x.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1)

        x = torch.cat((cls_tokens, x), dim=1)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights




class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        print(config.transformer["num_layers"])
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights




class Time_Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Time_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        print(config.transformer["num_layers"])
        for _ in range(4):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class Space_Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Space_Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        print(config.transformer["num_layers"])
        for _ in range(4):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights

class Transformer(nn.Module):
    def __init__(self, config, img_size, vis):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = Encoder(config, vis)

    def forward(self, input_ids, mask):
        embedding_output = self.embeddings(input_ids, mask)   #input_ids(1, 3, 224, 224)  embedding_output(1, 197, 768)
        encoded, attn_weights = self.encoder(embedding_output)  #encoded(1, 197, 768)   attn_weights:list 12*(1,12,197,197)
        return encoded, attn_weights

class Sequence_Transformer(nn.Module):
    def __init__(self, config, vis):
        super(Sequence_Transformer, self).__init__()
        # self.embeddings = Sequence_Embeddings(config)
        # self.encoder = Encoder(config, vis)

        self.embeddings_1 = Time_Sequence_Embeddings(config)
        self.encoder_1 = Time_Encoder(config, vis)
        self.embeddings_2 = Space_Sequence_Embeddings(config)
        self.encoder_2 = Space_Encoder(config, vis)
        self.batch_norm_1 = nn.BatchNorm1d(num_features=12)
        self.batch_norm_2 = nn.BatchNorm1d(num_features=12)



    def forward(self, x, LGE_key):
        batch_size = x.shape[0]
        x = x.view(batch_size, 12, 25, 25, 768)
        x = x.view(batch_size, 12, 625, 768)
        LGE_key = LGE_key.repeat(1,1,25,1)

        x_list = list(torch.split(x, 1))
        LGE_key_list = list(torch.split(LGE_key, 1))
        for index, value in enumerate(LGE_key_list):
            if torch.equal(value,torch.zeros_like(value)):
                print("torch.equal(value,torch.zeros_like(value))**************************")
                LGE_key_list[index] = x_list[index]
        LGE_key = torch.cat(LGE_key_list, dim=0)

        space_input_1 = []
        space_input_2 = []

        for i in range(12):
            y = x[:,i,:,:]
            y_2 = LGE_key[:,i,:,:]
            y = self.embeddings_1(y)  # input_ids(1, 3, 224, 224)  embedding_output(1, 197, 768)
            key_, _ = self.encoder_1(y)
            space_input_1.append(key_[:,0:1,:])
            y_2 = self.embeddings_1(y_2)  # input_ids(1, 3, 224, 224)  embedding_output(1, 197, 768)
            key_, _ = self.encoder_1(y_2)
            space_input_2.append(key_[:, 0:1, :])


        classtoken_input_1 = torch.cat(space_input_1, dim=1)
        classtoken_input_2 = torch.cat(space_input_2, dim=1)



        classtoken_input_1 = self.batch_norm_1(classtoken_input_1)
        classtoken_input_2 = self.batch_norm_2(classtoken_input_2)


        input_2 = torch.cat([classtoken_input_1, classtoken_input_2], dim=1)
        embedding_output = self.embeddings_2(input_2)
        encoded, attn_weights = self.encoder_2(embedding_output)
        return encoded, attn_weights






class DSFI(nn.Module):
    def __init__(self, config, structure ,num_classes=2, zero_head=False, vis=False):
        super(DSFI, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier

        self.transformer = Sequence_Transformer(config, vis)


        self.head = Linear(config.hidden_size, num_classes)

    def forward(self, x,LGE_key, labels=None):



        x, attn_weights = self.transformer(x, LGE_key)
        logits = self.head(x[:, 0])

        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_classes), labels.view(-1))
            return loss
        else:
            return logits, attn_weights





CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'sequence': configs.get_sequence_config()
}
