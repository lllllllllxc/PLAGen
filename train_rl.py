import os

import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import  build_plateau_optimizer
from modules.trainer_rl import Trainer
from modules.loss import RewardCriterion
from models.models import XProNet
import random



def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='../XproNet/data/iu_xray/resnet34_300/images300_seg/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='../XproNet/data/iu_xray/annotation.json', help='the path to the directory containing the data.')
    parser.add_argument('--label_path', type=str, default='../XproNet/data/iu_xray/labels/labels_14.pickle',
                        help='the path to the directory containing the data.')
    parser.add_argument('--exp_name', type=str, default='XPRONet',help='the name of the experiments.')

    parser.add_argument('--img_init_protypes_path', type=str, default='../XproNet/data/iu_xray/init_protypes_512.pt',
                        help='the path to the directory containing the data.')
    parser.add_argument('--init_protypes_path', type=str, default='../XproNet/data/iu_xray/init_prototypes.pt',
                        help='the path to the directory containing the data.')

    parser.add_argument('--text_init_protypes_path', type=str, default='../XproNet/data/iu_xray/text_empty_initprotypes_512.pt',
                        help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', choices=['iu_xray', 'mimic_cxr'], help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=60, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=6, help='the number of samples for a batch')#16

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='resnet101', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')
    parser.add_argument('--num_labels', type=int, default=14, help='the size of the label set')

    parser.add_argument('--d_txt_ebd', type=int, default=2048, help='the dimension of extracted text embedding.')
    parser.add_argument('--d_img_ebd', type=int, default=768, help='the dimension of extracted img embedding.')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=2048, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    # for Cross-modal Memory
    parser.add_argument('--topk', type=int, default=15, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')

    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=250, help='the number of training epochs.')#100
    parser.add_argument('--save_dir', type=str, default='results/iu_xray', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records_COMG/', help='the patch to save the results of experiments.')
    parser.add_argument('--log_period', type=int, default=100, help='the logging interval (in batches).')#10
    parser.add_argument('--save_period', type=int, default=1, help='the saving period (in epochs).')
    parser.add_argument('--sc_eval_period', type=int, default=10000, help='the saving period (in epochs).')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank in DDP.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=1e-6, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-5, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--adam_betas', type=tuple, default=(0.9, 0.98), help='the weight decay.')
    parser.add_argument('--adam_eps', type=float, default=1e-9, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')
    parser.add_argument('--noamopt_warmup', type=int, default=5000, help='.')
    parser.add_argument('--noamopt_factor', type=int, default=1, help='.')

    parser.add_argument('--reduce_on_plateau_factor', type=float, default=0.5, help='')
    parser.add_argument('--reduce_on_plateau_patience', type=int, default=3, help='')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=1, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.8, help='the gamma of the learning rate scheduler.')

    # Self-Critical Training
    parser.add_argument('--train_sample_n', type=int, default=1, help='The reward weight from cider')
    parser.add_argument('--train_sample_method', type=str, default='sample', help='')
    parser.add_argument('--train_beam_size', type=int, default=1, help='')
    parser.add_argument('--sc_sample_method', type=str, default='greedy', help='')
    parser.add_argument('--sc_beam_size', type=int, default=1, help='')

    # Others
    # 将默认值设置为随机整数
    random_seed = random.randint(0,30000)
    parser.add_argument('--seed', type=int, default=random_seed, help='Random seed.')
    parser.add_argument('--resume', type=str, default='../XproNet/results/iu_xray/XPRONet/model_best.pth', help='whether to resume the training from existing checkpoints.')

    parser.add_argument('--img_num_protype', type=int, default=10, help='.')
    parser.add_argument('--text_num_protype', type=int, default=10, help='.')
    parser.add_argument('--gbl_num_protype', type=int, default=10, help='.')
    parser.add_argument('--num_protype', type=int, default=20, help='.')
    parser.add_argument('--num_cluster', type=int, default=14, help='.')
    parser.add_argument('--img_con_margin', type=float, default=0.4, help='.')
    parser.add_argument('--txt_con_margin', type=float, default=0.4, help='.')

    args = parser.parse_args()
    return args


def main():
    # parse arguments
    args = parse_agrs()

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=True, drop_last=True)
    val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model = XProNet(args, tokenizer)

    # build optimizer, learning rate scheduler
    ve_optimizer, ed_optimizer = build_plateau_optimizer(args, model)

    # get function handles of loss and metrics
    criterion = RewardCriterion()
    metrics = compute_scores

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, ve_optimizer, ed_optimizer, args, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()


if __name__ == '__main__':
    main()
