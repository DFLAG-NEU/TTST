from __future__ import absolute_import, division, print_function
import os
import logging
import argparse
from tqdm import tqdm
import torch
import numpy as np
from sklearn.metrics import roc_auc_score

from models.modeling import CONFIGS, DSFI
from models.IEPV import IEPV
from models import vit
from util.data_utils import get_loader
from settings.defaults import _C
from settings.setup_functions import *

logger = logging.getLogger(__name__)

backbone = {
    'ViT-B_16': vit.get_b16_config()
}


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def setup_img(args):
    config = _C.clone()
    cfg_file = os.path.join('configs', 'MACE.yaml')
    config = SetupConfig(config, cfg_file)

    config.defrost()
    config.cuda_visible = '0'
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible

    config.nprocess, config.local_rank = SetupDevice()
    config.data.data_root, config.data.batch_size = LocateDatasets(config)

    config.freeze()
    SetSeed(config)

    num_classes = 2
    structure = vit.get_b16_config()

    model = IEPV(structure, config.data.img_size, num_classes)
    model.to(args.device)

    return args, model, structure


def model_sequence_setup(args, structure):

    sequence_config = CONFIGS['sequence']

    model = DSFI(sequence_config,
                 structure,
                 num_classes=2,
                 zero_head=True,
                 vis=True)

    model.to(args.device)

    return model


def valid(args, model_img, model_img_LGE, model, test_loader):

    eval_losses = AverageMeter()

    model_img.eval()
    model_img_LGE.eval()
    model.eval()

    all_preds = []
    all_label = []
    all_logits = []

    loss_fct = torch.nn.CrossEntropyLoss()

    epoch_iterator = tqdm(
        test_loader,
        desc="Testing...",
        bar_format="{l_bar}{r_bar}",
        dynamic_ncols=True
    )

    for step, batch in enumerate(epoch_iterator):

        batch = tuple(t.to(args.device) for t in batch)

        x, mask, LGE_images, LGE_masks, y = batch

        sequence_length = x.shape[1]
        LGE_length = LGE_images.shape[1]

        img_features = []
        keys = []
        positions = []

        with torch.no_grad():

            # Cine sequence
            for i in range(sequence_length):

                x_ = x[:, i, :, :, :]
                m_ = mask[:, i, :, :, :]

                _, img_feature_, key_, position_ = model_img(
                    x_, m_, test_mode=True
                )

                img_features.append(img_feature_)
                keys.append(key_)
                positions.append(position_)

            img_feature = torch.stack(img_features, dim=1)
            key = torch.stack(keys, dim=1)
            position = torch.stack(positions, dim=1)

            # LGE
            LGE_key = []

            for i in range(LGE_length):

                LGE_ = LGE_images[:, i, :, :, :]
                LGE_mask_ = LGE_masks[:, i, :, :, :]

                _, _, key_, _ = model_img_LGE(
                    LGE_, LGE_mask_, test_mode=True
                )

                LGE_key.append(key_)

            LGE_key = torch.stack(LGE_key, dim=1)

            logits = model(key, LGE_key)[0]

            loss = loss_fct(logits, y)

            eval_losses.update(loss.item())

            preds = torch.argmax(logits, dim=-1)

        if len(all_preds) == 0:

            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

        else:

            all_preds[0] = np.append(
                all_preds[0],
                preds.detach().cpu().numpy(),
                axis=0
            )

            all_label[0] = np.append(
                all_label[0],
                y.detach().cpu().numpy(),
                axis=0
            )

            all_logits[0] = np.append(
                all_logits[0],
                logits.detach().cpu().numpy(),
                axis=0
            )

    all_preds, all_label, all_logits = all_preds[0], all_label[0], all_logits[0]

    accuracy = simple_accuracy(all_preds, all_label)

    all_logits = all_logits[:, 0] - all_logits[:, 1]
    all_label = 1 - all_label

    auc = roc_auc_score(all_label, all_logits)

    logger.info("Test Loss: %.5f", eval_losses.avg)
    logger.info("Test Accuracy: %.5f", accuracy)
    logger.info("Test AUC: %.5f", auc)

    print("\n========== Test Result ==========")
    print("Accuracy:", accuracy)
    print("AUC:", auc)
    print("=================================\n")


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="MACE")

    parser.add_argument("--model_img_checkpoints_dir",
                        nargs='+',
                        default=[
                            "output/Cine_img_model.pth",
                            "output/LGE_image_model.pth"
                        ])

    parser.add_argument("--sequence_checkpoint",
                        default="output/TTST_SequenceTraining_checkpoint.pth")

    parser.add_argument("--train_data_folder",
                        default=r"D:\DataSet\train")

    parser.add_argument("--test_data_folder",
                        default=r"D:\DataSet\test")

    parser.add_argument("--eval_batch_size",
                        default=1,
                        type=int)

    parser.add_argument("--img_size",
                        default=224,
                        type=int)

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        level=logging.INFO
    )

    logger.warning("device: %s", device)

    # image encoder
    args, model_img, structure = setup_img(args)

    state_dict = torch.load(args.model_img_checkpoints_dir[0])
    model_img.load_state_dict(state_dict['model'])

    args, model_img_LGE, structure = setup_img(args)

    state_dict = torch.load(args.model_img_checkpoints_dir[1])
    model_img_LGE.load_state_dict(state_dict['model'])

    # sequence model
    model_sequence = model_sequence_setup(args, structure)

    seq_ckpt = torch.load(args.sequence_checkpoint)
    model_sequence.load_state_dict(seq_ckpt['model'])

    # dataset
    _, test_loader = get_loader(args)

    # run test
    valid(args, model_img, model_img_LGE, model_sequence, test_loader)


if __name__ == "__main__":
    main()