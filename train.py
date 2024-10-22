import argparse
import logging
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import random
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from networks.LGHF import LGHF
from trainer import trainer_synapse
import warnings

warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='path/to/train_npz', help='root dir for data')
parser.add_argument('--test_path', type=str,
                    default='path/to/test_vol_h5', help='root dir for data')
parser.add_argument('--dataset', type=str,
                    default='Synapse', help='experiment_name')
parser.add_argument('--list_dir', type=str,
                    default='path/to/lists/lists_Synapse', help='list dir')
parser.add_argument('--dst_fast', action='store_true',
                    help='SynapseDatasetFast: will load all data into RAM')
parser.add_argument('--num_classes', type=int,
                    default=9, help='output channel of network')
parser.add_argument('--n_skip_bridge', type=int,
                    default=1, help='Number of skip connections (repeat the skip connection $n$ times)')
parser.add_argument('--pyramid_levels', type=int,
                    default=4, help='Number of pyramid levels')
parser.add_argument('--model_path', type=str,
                    default='./model_out/best_model.pth', help='model path')
parser.add_argument('--resume', action='store_true', help='resume from checkpoint')

parser.add_argument('--output_dir', type=str, 
                    default='./model_out',help='output dir')                   
parser.add_argument('--max_iterations', type=int,
                    default=90000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int,
                    default=600, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=20, help='batch_size per gpu')
parser.add_argument('--num_workers', type=int,
                    default=4, help='num_workers')
parser.add_argument('--eval_interval', type=int,
                    default=20, help='eval_interval')
parser.add_argument('--model_name', type=str,
                    default='LGHF_2DES', help='model_name')
parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')
parser.add_argument('--z_spacing', type=int,
                    default=1, help='z_spacing')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')

args = parser.parse_args()


args.output_dir = args.output_dir + f'/{args.model_name}'
os.makedirs(args.output_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


if __name__ == "__main__":
    
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    dataset_name = args.dataset

    if args.batch_size != 24 and args.batch_size % 5 == 0:
        args.base_lr *= args.batch_size / 24

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    net = LGHF(num_classes=args.num_classes, n_skip_bridge=args.n_skip_bridge,
                        pyramid_levels=args.pyramid_levels).to(device)

    trainer = {'Synapse': trainer_synapse,}
    trainer[dataset_name](args, net, args.output_dir)
