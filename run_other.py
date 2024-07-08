import argparse
import logging
import os
import random

import numpy as np
import torch
import torch as th
import pytorch_lightning as pl

import config
import helpers
from data.data_module import DataModule
from torch.utils.data import Subset, TensorDataset

from models.SEM import Network
from models.SEM_trainer import sem_train
from models.build_model import build_model


import warnings

from trainer import GAN_Attack

warnings.filterwarnings("ignore", category=UserWarning)

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def load_weights(model, tag, weights_file):

    loaded_state_dict = th.load(weights_file)

    missing, unexpected = model.load_state_dict(loaded_state_dict, strict=False)


    print(f"Successfully loaded initial weights from {weights_file}")
    if missing:
        print(f"Weights {missing} were not present in the initial weights file.")
    if unexpected:
        print(f"Unexpected weights {unexpected} were present in the initial weights file. These will be ignored.")

def eamc_train(cfg, net, train_loder, test_loader):

    enc_opt, dis_opt = net.configure_optimizers()  # 

    for e in range(cfg.n_epochs):
        loss_tot_all = 0
        net.train()
        for idx, batch in enumerate(train_loder):
            views = batch[:-1]
            labels = batch[-1]
            views = [tmp.to(net.device) for tmp in views]
            # views_ad = PGD_contrastive(net, views, eps=0.15, alpha=0.02)  # 

            enc_opt.zero_grad()
            loss_tot = net.training_step(views+[labels], idx, 0)  # 

            loss_tot.backward()
            enc_opt.step()

            dis_opt.zero_grad()
            loss_tot = net.training_step(views + [labels], idx, 1)  # 

            loss_tot.backward()
            dis_opt.step()

            loss_tot_all += loss_tot.detach().data
        logger.info('epoch:{}, loss:{:.4f}'
              .format(e+1, loss_tot_all.data))
        if e % 1 == 0:
            with torch.no_grad():
                net.eval()
                test_list = []
                for idx, batch in enumerate(test_loader):
                    batch = [item.to(net.device) for item in batch]
                    test_list.append(net._val_test_step(batch, idx, 'test'))
                mtc = net._val_test_epoch_end(test_list, 'test')
                logger.info('valid real metric:acc:{:4f} nmi:{:4f} ari:{:4f}'.format(mtc['acc'], mtc['nmi'], mtc['ari']))

            torch.save(net.state_dict(), str(args.save_dir) + '/{}.ckpt'.format(e))
            logger.info('model has been saved in {}'.format(str(args.save_dir) + '/{}.ckpt'.format(e)))

def ori_train(cfg, net, train_loder, test_loader, log_loss=True):

    optimizer = net.configure_optimizers()  # 

    for e in range(cfg.n_epochs):
        loss_tot_all = 0
        net.train()
        for idx, batch in enumerate(train_loder):
            views = batch[:-1]
            labels = batch[-1]
            views = [tmp.to(net.device) for tmp in views]

            optimizer.zero_grad()
            loss_tot = net.training_step(views+[labels], idx)  # 

            loss_tot.backward()
            optimizer.step()

            loss_tot_all += loss_tot.detach().data
        if log_loss:
            logger.info('epoch:{}, loss:{:.4f}' .format(e+1, loss_tot_all.data))
        if e % 1 == 0:
            with torch.no_grad():
                net.eval()
                test_list = []
                for idx, batch in enumerate(test_loader):
                    batch = [item.to(net.device) for item in batch]
                    test_list.append(net._val_test_step(batch, idx, 'test'))
                mtc = net._val_test_epoch_end(test_list, 'test')
                logger.info('real data metric:acc:{:4f} nmi:{:4f} ari:{:4f}'.format(mtc['acc'], mtc['nmi'], mtc['ari']))

            torch.save(net.state_dict(), str(args.save_dir) + '/{}.ckpt'.format(e))
            logger.info('model has been saved in {}'.format(str(args.save_dir) + '/{}.ckpt'.format(e)))


def pre_main(ename, cfg, args, save_event):

    set_seed(5)

    logger.info(ename)

    n = int(args.percent * args.num)
    data_module = DataModule(cfg.dataset_config)


    if 'patchedmnist' in ename:
        trainset = data_module.train_dataset[:]
        trainset = [trainset[i] for i in [0, 1, 3, -1]]
        data_module.train_dataset = TensorDataset(*trainset)


    data_module.train_dataset = TensorDataset(*(data_module.train_dataset[:n]))

    logger.info('{} used {} samples'.format(ename, len(data_module.train_dataset)))

    train_loader = data_module.train_dataloader(shuffle=True, drop_last=True)
    test_loader = data_module.train_dataloader(shuffle=True, drop_last=False)

    if 'SEM' in ename:
        net = Network(cfg.n_views, args.dims, 512, 128, cfg.n_clusters, config.DEVICE)
        net.to(args.device)
    else:
        net = build_model(cfg.model_config)
    print(net)

    save_dir = helpers.get_save_dir(ename, save_event, 0)
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir

    if 'SEM' not in ename:
        if net.requires_pre_train:
            if ename not in ['noisymnist_mimvc', 'noisyfashionmnist_mimvc']:
                net.init_pre_train()
                logger.info('begin net pretraining!!!')
                cfg_ = cfg
                cfg_.n_epochs = 20
                ori_train(cfg, net, train_loader, test_loader)
                net.init_fine_tune()

    if 'eamc' in ename:
        eamc_train(cfg, net, train_loader, test_loader)
    elif 'SEM' in ename:
        sem_train(args.data_name, net, train_loader, test_loader, logger)
    elif 'mvae' in ename:
        ori_train(cfg, net, train_loader, test_loader, log_loss=False)
    else:
        ori_train(cfg, net, train_loader, test_loader)


    torch.save(net.state_dict(), str(save_dir)+'/best.ckpt')
    logger.info('model has been saved in {}'.format(str(save_dir)+'/best.ckpt'))
    return

def atk_main(ename, cfg, args, tag):

    set_seed(5)

    data_module = DataModule(cfg.dataset_config)
    n = int(args.percent * args.num)

    if 'patchedmnist' in ename:
        trainset = data_module.train_dataset[:]
        trainset = [trainset[i] for i in [0, 1, 3, -1]]
        data_module.train_dataset = TensorDataset(*trainset)

    data_module.train_dataset = TensorDataset(*(data_module.train_dataset[n:]))

    logger.info('{} used {} samples'.format(ename, len(data_module.train_dataset)))

    train_loader = data_module.train_dataloader(shuffle=True, drop_last=True)
    test_loader = data_module.train_dataloader(shuffle=True, drop_last=False)

    if 'SEM' in ename:
        tar_model = Network(cfg.n_views, args.dims, 512, 128, cfg.n_clusters, config.DEVICE)
        tar_model.to(args.device)
    else:
        tar_model = build_model(cfg.model_config)
    print(tar_model)

    save_dir = helpers.get_save_dir(ename, tag, run=0)
    load_weights(tar_model, tag, save_dir / 'best.ckpt')

    for param in tar_model.parameters():
        param.requires_grad = False


    trainer = GAN_Attack(args, tar_model, perb_eps=args.atk_eps)
    print(trainer.netGs)
    print(trainer.netDs)
    _, mtc_list = trainer.train(train_loader, test_loader, args.atk_epochs)

    atk_save_dir = helpers.get_atk_save_dir(ename, 'atk_gans', run=0)
    os.makedirs(atk_save_dir, exist_ok=True)
    torch.save(trainer.netGs.state_dict(), str(atk_save_dir) + '/best.ckpt')
    logger.info('atk_model has been saved in {}'.format(str(atk_save_dir) + '/best.ckpt'))

    return mtc_list

if __name__ == '__main__':
    print("Torch version:", th.__version__)
    print("Lightning version:", pl.__version__)
    from tools import log
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='eamc') # ['eamc', 'simvc', 'comvc', 'mvae', 'cae', 'mimvc', 'SEM']
    parser.add_argument('--data_name', default='noisymnist')
    args = parser.parse_args()

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

    args.percent = 0.5
    args.device = config.DEVICE
    args.atk_mode = 1 if args.model_name != 'mvae' else 2
    logger = log('logs', '{}_{}_{}'.format(args.model_name, args.data_name, int(args.percent * 100)), is_cover=True)

    logger.info(args)
    logger.info('pretraining total model!!!')
    ename, cfg = config.get_experiment_config(args.data_name, args.model_name)
    cfg.n_epochs = 20
    cfg.dataset_config.n_train_samples = args.num
    pmtc_list = pre_main(ename, cfg, args, 'pretrain')

    args.atk_epochs = 30
    logger.info('attacking pretraining model!!!')
    amtc_list = atk_main(ename, cfg, args, 'pretrain')

    logger.handlers.clear()
    logging.shutdown()
    print('logging shut down!')

