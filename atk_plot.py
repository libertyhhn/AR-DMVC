import argparse
import logging
import os
import random
import warnings

import numpy as np
import torch
import torch as th
import pytorch_lightning as pl
from torch.utils.data import Subset, TensorDataset

import config
import helpers
from data.data_module import DataModule
from models.build_model import build_model

from trainer import GAN_Attack
from tools import log
from models.SEM import Network

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def load_weights(model, tag, weights_file):

    loaded_state_dict = th.load(weights_file)

    missing, unexpected = model.load_state_dict(loaded_state_dict, strict=False)

    print(f"Successfully loaded initial weights from {weights_file}")
    if missing:
        print(f"Weights {missing} were not present in the initial weights file.")
    if unexpected:
        print(f"Unexpected weights {unexpected} were present in the initial weights file. These will be ignored.")

def main(ename, cfg, args, tag):

    # ename = data_name + '_' + model_name
    setup_seed(5)

    data_module = DataModule(cfg.dataset_config)
    n = int(args.percent * args.num)

    if 'patchedmnist' in ename:
        trainset = data_module.train_dataset[:]
        trainset = [trainset[i] for i in [0, 1, 3, -1]]
        data_module.train_dataset = TensorDataset(*trainset)

    data_module.train_dataset = TensorDataset(*(data_module.train_dataset[n:]))

    logger.info('{} used {} samples'.format(ename, len(data_module.train_dataset)))


    test_loader = data_module.train_dataloader(shuffle=True, drop_last=False)

    # 准备预训练目标模型
    if 'SEM' in ename:
        tar_model = Network(cfg.n_views, args.dims, 512, 128, cfg.n_clusters, config.DEVICE)
        tar_model.to(config.DEVICE)
    else:
        tar_model = build_model(cfg.model_config)
    # print(tar_model)
    save_dir = helpers.get_save_dir(ename, tag, run=0)
    load_weights(tar_model, tag, save_dir / 'best.ckpt')
    # 冻结模型
    for param in tar_model.parameters():
        param.requires_grad = False



    trainer = GAN_Attack(args, tar_model, perb_eps=args.atk_eps)
    atk_save_dir = helpers.get_atk_save_dir(ename, 'atk_gans', run=0)
    load_weights(trainer.netGs, 'atk_model', atk_save_dir / 'best.ckpt')

    trainer.val_real(test_loader)
    pred_labels = torch.tensor(trainer.val_fake(test_loader, is_get_pred=True)[1], dtype=torch.int64)
    data_module.train_dataset = TensorDataset(*(data_module.train_dataset[:]+(pred_labels,)))
    test_loader = data_module.train_dataloader(shuffle=False, drop_last=False)

    plot_save_dir = helpers.config.PROJECT_ROOT / 'atk_plot' / ename
    os.makedirs(plot_save_dir, exist_ok=True)
    trainer.plot_ad_images(test_loader, plot_save_dir)

    return


if __name__ == '__main__':
    print("Torch version:", th.__version__)
    print("Lightning version:", pl.__version__)

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ardmvc_am')  # ['eamc', 'simvc', 'comvc', 'mvae', 'cae', 'mimvc', 'SEM',  'adcomvc_nkl', 'adcomvc']
    parser.add_argument('--data_name', default='noisymnist')
    args = parser.parse_args()

    args.device = config.DEVICE


    args.percent = 0.5

    if args.data_name == 'noisymnist':
        args.num = 70000
        args.atk_eps = 0.3
        args.dims = [28 * 28, 28 * 28]
    elif args.data_name == 'noisyfashionmnist':
        args.num = 70000
        args.atk_eps = 0.15
        args.dims = [28 * 28, 28 * 28]
    elif args.data_name == 'patchedmnist':
        args.num = 21770
        args.atk_eps = 0.3
        args.dims = [28 * 28, 28 * 28, 28 * 28]


    ename, cfg = config.get_experiment_config(args.data_name, args.model_name)

    cfg.dataset_config.n_train_samples = args.num

    logger = log('atk_plot/logs', ename, is_cover=True)

    logger.info('Attack plot')

    logger.info(args)

    main(ename, cfg, args, 'pretrain')


    logger.handlers.clear()
    logging.shutdown()
    print('logging shut down!')

