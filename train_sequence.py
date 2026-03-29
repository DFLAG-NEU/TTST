from __future__ import absolute_import, division, print_function
import logging
import argparse
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from apex import amp
from models.modeling import CONFIGS, DSFI
from util.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from util.data_utils import get_loader
from models.IEPV import IEPV
from models import vit
from settings.defaults import _C
from settings.setup_functions import *
import time
from sklearn.metrics import roc_auc_score
import numpy as np

backbone = {
	'ViT-B_16': vit.get_b16_config()
}


logger = logging.getLogger(__name__)


class AverageMeter(object):
    """Computes and stores the average and current value"""
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


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.output_dir, "%s_checkpoint.pth" % args.name)
    torch.save({'model': model.state_dict()}, model_checkpoint)
    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)

def setup_img(args):
    config = _C.clone()
    cfg_file = os.path.join('configs', 'MACE.yaml')
    config = SetupConfig(config, cfg_file)
    config.defrost()
    ## Log Name and Perferences
    config.write = True  # comment it to disable all the log writing
    config.train.checkpoint = True  # comment it to disable saving the checkpoint
    config.misc.exp_name = f'{config.data.dataset}'
    config.misc.log_name = f'IELT'
    config.cuda_visible = '0,1,2,3'
    # Environment Settings
    config.data.log_path = os.path.join(config.misc.output, config.misc.exp_name, config.misc.log_name
                                        + time.strftime('-%m-%d_%H-%M', time.localtime()))

    config.model.pretrained = os.path.join(config.model.pretrained,
                                           config.model.name + config.model.pre_version + config.model.pre_suffix)
    os.environ['CUDA_VISIBLE_DEVICES'] = config.cuda_visible
    os.environ['OMP_NUM_THREADS'] = '1'
    # Setup Functions
    config.nprocess, config.local_rank = SetupDevice()
    config.data.data_root, config.data.batch_size = LocateDatasets(config)
    config.train.lr = ScaleLr(config)
    log = SetupLogs(config, config.local_rank)
    if config.write and config.local_rank in [-1, 0]:
        with open(config.data.log_path + '/config.json', "w") as f:
            f.write(config.dump())
    config.freeze()
    SetSeed(config)
    num_classes = 2
    structure = vit.get_b16_config()
    model = IEPV(structure, config.data.img_size, num_classes)
    model.to(args.device)
    return args, model, structure




def model_sequence_setup(args, structure):
    sequence_config = CONFIGS['sequence']
    model = DSFI(sequence_config, structure, num_classes=2, zero_head=True, vis=True)
    model.to(args.device)
    return model




def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)



def valid(args, model_img, model_img_LGE, model, writer, test_loader, global_step):
    # Validation!
    eval_losses = AverageMeter()
    logger.info("***** Running Validation *****")
    logger.info("  Num steps = %d", len(test_loader))
    logger.info("  Batch size = %d", args.eval_batch_size)
    model_img.eval()
    model_img_LGE.eval()
    model.eval()
    all_preds, all_label, all_logits = [], [], []
    epoch_iterator = tqdm(test_loader,
                          desc="Validating... (loss=X.X)",
                          bar_format="{l_bar}{r_bar}",
                          dynamic_ncols=True,
                          disable=False)
    loss_fct = torch.nn.CrossEntropyLoss()
    for step, batch in enumerate(epoch_iterator):
        torch.cuda.empty_cache()
        model_img.eval()
        model_img_LGE.eval()
        model.eval()
        batch = tuple(t.to(args.device) for t in batch)
        x, mask, LGE_images, LGE_masks, y = batch
        LGE_images_list = torch.split(LGE_images, 1)
        is_all_zero = []
        for i in LGE_images_list:
            YorN = torch.equal(i, torch.zeros_like(i))
            is_all_zero.append(YorN)
        with torch.no_grad():
            sequence_length = x.shape[1]
            LGE_length = LGE_images.shape[1]
            img_features = []
            keys = []
            positions = []
            for i in range(sequence_length):
                torch.cuda.empty_cache()
                x_ = x[:, i, :, :, :]
                m_ = mask[:, i, :, :, :]
                _, img_feature_, key_, position_ = model_img(x_,m_,test_mode = True)


                # img_feature.append(img_feature_)
                img_features.append(img_feature_)
                keys.append(key_)
                positions.append(position_)

            # img_feature = torch.stack(img_features, dim=1)
            img_feature = torch.stack(img_features, dim=1)
            key = torch.stack(keys, dim=1)
            position = torch.stack(positions, dim=1)
            LGE_key = []
            for i in range(LGE_length):
                LGE_ = LGE_images[:, i, :, :, :]
                LGE_mask_ = LGE_masks[:, i, :, :, :]
                _, _, key_, _ = model_img_LGE(LGE_, LGE_mask_, test_mode=True)
                LGE_key.append(key_)
            LGE_key = torch.stack(LGE_key, dim=1)


            treeD_feature = []
            for i in range(12):
                x_ = img_feature[:, 25*i:(25+1)*i, :]
                x_ = torch.sum(x_, dim=1)/25
                treeD_feature.append(x_)
            treeD_feature = torch.stack(treeD_feature, dim=1)
            mask = torch.zeros(x.shape[0], 12, 14 * 14, 768).to(args.device)
            new_tensor = torch.zeros(x.shape[0], 12, 14 * 14, 768).to(args.device)
            mask_weight = torch.ones(768).to(args.device)
            for slice in range(12):
                for i in range(x.shape[0]):
                    for j in range(slice * 25, (slice + 1) * 25):
                        for k in range(24):
                            new_tensor[i, slice, position[i, j, k] - 1] = new_tensor[i, slice, position[i, j, k] - 1] + \
                                                                          key[
                                                                              i, j, k + 1]
                            mask[i, slice, position[i, j, k] - 1] = mask[i, slice, position[i, j, k] - 1] + mask_weight
            new_tensor = torch.where(mask == 0, 0, new_tensor / mask)
            a, b, c, d = new_tensor.shape
            new_tensor = new_tensor.view(a, b * c, d)
            new_tensor = torch.cat([img_feature, new_tensor], dim=1)


            treeD_feature_list = list(torch.split(treeD_feature, 1))
            LGE_key_list = list(torch.split(LGE_key, 1))
            for index, value in enumerate(is_all_zero):
                if value:
                    LGE_key_list[index] = torch.zeros_like(LGE_key_list[index])
            LGE_key = torch.cat(LGE_key_list, dim=0)

            logits = model(key, LGE_key)[0]

            eval_loss = loss_fct(logits, y)
            eval_losses.update(eval_loss.item())

            preds = torch.argmax(logits, dim=-1)


        if len(all_preds) == 0:
            all_preds.append(preds.detach().cpu().numpy())
            all_label.append(y.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())
        else:
            all_preds[0] = np.append(
                all_preds[0], preds.detach().cpu().numpy(), axis=0
            )
            all_label[0] = np.append(
                all_label[0], y.detach().cpu().numpy(), axis=0
            )
            all_logits[0] = np.append(
                all_logits[0], logits.detach().cpu().numpy(), axis=0
            )
        epoch_iterator.set_description("Validating... (loss=%2.5f)" % eval_losses.val)

    all_preds, all_label, all_logits = all_preds[0], all_label[0], all_logits[0]
    accuracy = simple_accuracy(all_preds, all_label)
    all_logits = all_logits[:, 0] - all_logits[:, 1]
    all_label = 1 - all_label
    auc = roc_auc_score( all_label, all_logits)
    logger.info("\n")
    logger.info("Validation Results")
    logger.info("Global Steps: %d" % global_step)
    logger.info("Valid Loss: %2.5f" % eval_losses.avg)
    logger.info("Valid Accuracy: %2.5f" % accuracy)
    logger.info("Valid AUC: %2.5f" % auc)

    writer.add_scalar("test/accuracy", scalar_value=accuracy, global_step=global_step)
    writer.add_scalar("test/AUC", scalar_value=auc, global_step=global_step)
    return accuracy,auc


def train(args, model_img,model_img_LGE, model):
    model_img.train()
    model_img_LGE.train()
    model.train()
    """ Train the model """

    os.makedirs(args.output_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=os.path.join("logs", args.name))

    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps

    # Prepare dataset
    train_loader, test_loader = get_loader(args)

    optimizer = torch.optim.Adam(model.parameters(),
                                lr=args.learning_rate,
                                weight_decay=args.weight_decay)
    t_total = args.num_steps
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    if args.fp16:
        model, optimizer = amp.initialize(models=model,
                                          optimizers=optimizer,
                                          opt_level=args.fp16_opt_level)
        amp._amp_state.loss_scalers[0]._loss_scale = 2**20


    # Train!
    logger.info("***** Running training *****")
    logger.info("  Total optimization steps = %d", args.num_steps)
    logger.info("  Instantaneous batch size per GPU = %d", args.train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps)
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)

    model.zero_grad()
    set_seed(args)  # Added here for reproducibility (even between python 2 and 3)
    losses = AverageMeter()
    global_step, best_acc, best_auc = 0, 0, 0
    while True:
        epoch_iterator = tqdm(train_loader,
                              desc="Training (X / X Steps) (loss=X.X)",
                              bar_format="{l_bar}{r_bar}",
                              dynamic_ncols=True,
                              disable=False)
        for step, batch in enumerate(epoch_iterator):
            torch.cuda.empty_cache()
            model_img.train()
            model_img_LGE.train()
            model.train()
            batch = tuple(t.to(args.device) for t in batch)
            x, mask ,LGE_images, LGE_masks, y = batch
            LGE_images_list = torch.split(LGE_images, 1)
            is_all_zero = []
            for i in LGE_images_list:
                YorN = torch.equal(i, torch.zeros_like(i))
                is_all_zero.append(YorN)
            print(is_all_zero)
            sequence_length = x.shape[1]
            LGE_length = LGE_images.shape[1]
            img_features = []


            with torch.no_grad():
                for i in range(sequence_length):
                    torch.cuda.empty_cache()
                    x_ = x[:,i,:,:,:]
                    m_ = mask[:,i,:,:,:]
                    _, img_features_= model_img(x_,m_,test_mode = True)
                    img_features.append(img_features_)
                Cine_feature = torch.stack(img_features, dim=1)


                LGE_feature = []
                for i in range(LGE_length):
                    LGE_ = LGE_images[:,i,:,:,:]
                    LGE_mask_ = LGE_masks[:,i,:,:,:]
                    _, img_features_ = model_img_LGE(LGE_,LGE_mask_,test_mode = True)
                    LGE_feature.append(img_features_)
                LGE_feature = torch.stack(LGE_feature, dim=1)



            LGE_feature_list = list(torch.split(LGE_feature, 1))
            for index, value in enumerate(is_all_zero):
                if value:
                    LGE_feature_list[index] = torch.zeros_like(LGE_feature_list[index])
            LGE_feature = torch.cat(LGE_feature_list, dim=0)



            loss = model(Cine_feature, LGE_feature ,y)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
            else:
                loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                losses.update(loss.item()*args.gradient_accumulation_steps)
                if args.fp16:
                    torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scheduler.step()
                optimizer.step()
                optimizer.zero_grad()
                global_step += 1

                epoch_iterator.set_description(
                    "Training (%d / %d Steps) (loss=%2.5f)" % (global_step, t_total, losses.val)
                )
                writer.add_scalar("train/loss", scalar_value=losses.val, global_step=global_step)
                writer.add_scalar("train/lr", scalar_value=scheduler.get_lr()[0], global_step=global_step)

                if global_step % args.eval_every == 0 :
                    torch.cuda.empty_cache()
                    with torch.no_grad():
                        accuracy, auc= valid(args, model_img, model_img_LGE, model, writer, test_loader, global_step)
                    model_checkpoint = os.path.join(args.output_dir, "%s%d_checkpoint.pth" % (args.name, global_step))
                    # torch.save({'model': model.state_dict()}, model_checkpoint)
                    logger.info("Saved model checkpoint to [DIR: %s]", args.output_dir)
                    # if best_acc < accuracy:
                    #     save_model(args, model)
                    #     best_acc = accuracy
                    if best_auc < auc:
                        save_model(args, model)
                        best_auc = auc
                    model.train()

                if global_step % t_total == 0:
                    break
        losses.reset()
        if global_step % t_total == 0:
            break


    writer.close()
    logger.info("Best AUC: \t%f" % auc)
    logger.info("End Training!")


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", type=str, default="TTST_SequenceTraining",
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", type=str, default="MACE")
    parser.add_argument("--model_img_checkpoints_dir", type=str, nargs='+',
                        default=["output\Cine_img_model.pth", "output\LGE_image_model.pth"],
                        help='List of model image checkpoint files')

    parser.add_argument("--train_data_folder", type=str, default=r"D:\DataSet\train")
    parser.add_argument("--test_data_folder", type=str, default=r"D:\DataSet\test")

    parser.add_argument("--model_type", type=str ,default="ViT-B_16")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=224, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=2, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=1, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_every", default=50, type=int)

    parser.add_argument("--learning_rate", default=1e-4, type=float,
                        help="The initial learning rate.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=3000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=50, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed', type=int, default=0,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O2',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument('--loss_scale', type=float, default=0,
                        help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                             "0 (default value): dynamic loss scaling.\n"
                             "Positive power of 2: static loss scaling value.\n")
    args = parser.parse_args()


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)
    logger.warning(" device: %s, 16-bits training: %s" %
                   ( args.device,  args.fp16))

    # Set seed
    set_seed(args)

    # Model & Tokenizer Setup
    args, model_img, structure = setup_img(args)
    state_dict = torch.load(args.model_img_checkpoints_dir[0])
    model_img.load_state_dict(state_dict['model'])
    args, model_img_LGE, structure = setup_img(args)
    state_dict = torch.load(args.model_img_checkpoints_dir[1])
    model_img_LGE.load_state_dict(state_dict['model'])

    model_sequence = model_sequence_setup(args, structure)

    # Training
    train(args, model_img, model_img_LGE, model_sequence)


if __name__ == "__main__":
    main()
