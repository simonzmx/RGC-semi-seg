import random
import numpy as np
import torch
import yaml
from argparse import ArgumentParser, Namespace
from pathlib import Path
from nns import training_pipeline


# Can change the parameters here
parser = ArgumentParser()

"""Overall Parameters"""
parser.add_argument('--data_cfg', type=Path, default='./cfg/data/dataset_iu.yaml',
                    help='data configuration file path')
parser.add_argument('--model_cfg', type=Path, default='./cfg/model/hyp.semi_seg_cps_iu.yaml',
                    help='model configuration file path')

"""Running Options"""
parser.add_argument('--batch_size', type=int, default=2, help='batch size for training')
parser.add_argument('--no_train', action='store_true', help='not run training')
parser.add_argument('--no_test', action='store_true', help='not run testing')
parser.add_argument('--resume_training', action='store_true', help='load checkpoint from previous training and resume')
parser.add_argument('--cpu', action='store_true', help='force using cpu, otherwise use gpu when available')

"""Visualizations"""
parser.add_argument('--visualize_prob_map', action='store_true',
                    help='visualize the output probability maps from model')

"""Logging"""
parser.add_argument('--exp_name', type=str, default=None, help='experiment name for logging')

args = parser.parse_args()


def main(main_args: Namespace) -> None:
    # Load data settings from the yaml file
    with open(main_args.data_cfg, 'rb') as f:
        data_cfg = yaml.load(f, Loader=yaml.SafeLoader)
    # Load hyper-parameters from the yaml file
    with open(main_args.model_cfg, 'rb') as f:
        hyp = yaml.load(f, Loader=yaml.SafeLoader)

    print(main_args)
    print("Data configuration:")
    print(data_cfg)
    print("Model hyper-parameters:")
    print(hyp)
    hyp.update(data_cfg)

    if 'random_seed' in hyp and hyp['random_seed'] is not None:
        random_seed = int(hyp['random_seed'])
    else:
        random_seed = 1104
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)

    training_pipeline.process(main_args, data_cfg, hyp)


if __name__ == '__main__':
    main(args)
