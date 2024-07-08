import argparse
import logging
import os
import random

import numpy as np
import scipy
import torch
import torch as th

import config
import helpers
from data.data_module import DataModule
from torch.utils.data import TensorDataset
from models.build_model import build_model

import torch.nn.functional as Func

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

def PGD_contrastive(net, views, eps=8. / 255., alpha=2. / 255., iters=10):  # eps=8. / 255., alpha=2. / 255., iters=10
    # init
    deltas = []
    for x in views:
        delta = torch.rand_like(x) * eps * 2 - eps
        delta = torch.nn.Parameter(delta)
        deltas.append(delta)


    for i in range(iters):

        net.zero_grad()
        net([x + d for x, d in zip(views, deltas)])  
        losses = net.get_loss()  # 求loss  DDC1 DDC2 DDC3 Contrast/01 MSE/0 MSE/1 tot
        losses['Contrast/01'].backward()
        # print("loss is {}".format(loss))

        for v in range(len(views)):
            deltas[v].data = deltas[v].data + alpha * deltas[v].grad.sign()
            deltas[v].grad = None
            deltas[v].data = torch.clamp(deltas[v].data, min=-eps, max=eps)
            deltas[v].data = torch.clamp(views[v] + deltas[v].data, min=0, max=1) - views[v]

            # d.data = d.data + alpha * d.grad.sign()
            # d.grad = None
            # d.data = torch.clamp(d.data, min=-eps, max=eps)
            # d.data = torch.clamp(x + d.data, min=0, max=1) - x

    return [(x + d).detach() for x, d in zip(views, deltas)]


def train(args, cfg, net, train_loder, test_loader):

    optimizer = net.configure_optimizers()  # 
    mtc_list = []
    for e in range(cfg.n_epochs):
        loss_tot_all, loss_ori_all, loss_ad_all, loss_kl_all = 0, 0, 0, 0
        net.train()
        for idx, batch in enumerate(train_loder):
            views = batch[:-1]
            labels = batch[-1]
            views = [tmp.to(net.device) for tmp in views]
            views_ad = PGD_contrastive(net, views, eps=args.eps, alpha=args.alpha)  # eps=0.15 alpha=0.02

            optimizer.zero_grad()
            loss_ori = net.training_step(views+[labels], idx)  # 
            P1 = net.output
            loss_ad = net.training_step(views_ad+[labels], idx)
            P2 = net.output

            eps = 0.0001 * torch.ones_like(P1, requires_grad=False).to(P1.device)
            P1 = P1 + eps
            P2 = P2 + eps
            loss_kl = Func.kl_div(P2.log(), P1, reduction='batchmean')


            loss_tot = loss_ori + args.para_ad * loss_ad + args.para_kl * loss_kl

            loss_tot.backward()
            optimizer.step()

            loss_ori_all += loss_ori.detach().data
            loss_ad_all += loss_ad.detach().data
            loss_kl_all += loss_kl.detach().data
            loss_tot_all += loss_tot.detach().data
        logger.info('epoch:{}, loss_ori:{:.4f}, loss_ad:{:.4f}, loss_kl:{:4f}, loss:{:.4f}'
              .format(e+1, loss_ori_all.data, loss_ad_all.data, loss_kl_all.data, loss_tot_all.data))
        if e % 1 == 0:
            with torch.no_grad():
                net.eval()
                test_list = []
                for idx, batch in enumerate(test_loader):
                    batch = [item.to(net.device) for item in batch]
                    test_list.append(net._val_test_step(batch, idx, 'test'))
                mtc = net._val_test_epoch_end(test_list, 'test')
                logger.info('real data metric :acc:{:4f} nmi:{:4f} ari:{:4f}'.format(mtc['acc'], mtc['nmi'], mtc['ari']))
                torch.save(net.state_dict(), str(args.save_dir) + '/epoch{}.ckpt'.format(e+1))
                logger.info('model has been saved in {}/epoch{}.ckpt!'.format(str(args.save_dir), e+1))

                mtc_list.append(np.array([mtc['acc'], mtc['nmi'], mtc['ari']]))

    return mtc, mtc_list


def pre_main(ename, cfg, args, tag):

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

    net = build_model(cfg.model_config)
    print(net)

    save_dir = helpers.get_save_dir(ename, tag, 0)
    os.makedirs(save_dir, exist_ok=True)
    args.save_dir = save_dir
    _, mtc_list = train(args, cfg, net, train_loader, test_loader)

    torch.save(net.state_dict(), str(save_dir)+'/best.ckpt')
    logger.info('model has been saved in {}'.format(str(save_dir)+'/best.ckpt'))

    return mtc_list

def atk_main(ename, cfg, args, tag):

    # ename = data_name + '_' + model_name
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

    from tools import log
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='ardmvc_am')
    parser.add_argument('--data_name', default='noisymnist')
    args = parser.parse_args()

    if 'ardmvc' not in args.model_name:
        raise ValueError("Model name is not right.")

    args.eps = 0.2
    args.alpha = 0.02
    args.percent = 0.5


    args.device = config.DEVICE
    args.atk_mode = 1


    if args.data_name == 'noisymnist':  # in ['noisymnist', 'noisyfashionmnist']:
        args.num = 70000
        args.atk_eps = 0.3
        args.para_ad = 1
        args.para_kl = 0.1
        args.epochs = 30
    elif args.data_name == 'noisyfashionmnist':
        args.num = 70000
        args.atk_eps = 0.15
        args.para_ad = 0.1
        args.para_kl = 1  #
        args.epochs = 30
    elif args.data_name == 'patchedmnist':
        args.num = 21770
        args.atk_eps = 0.3
        args.para_ad = 0.1
        args.para_kl = 0.1
        args.epochs = 40


    if args.model_name == 'ardmvc':
        args.para_kl = 0
    logger = log('logs', '{}_{}_{}'.format(args.model_name, args.data_name, int(0.5*100)), is_cover=True)

    logger.info(args)

    logger.info('pretraining total model!!!')
    ename, cfg = config.get_experiment_config(args.data_name, args.model_name)
    cfg.n_epochs = args.epochs
    cfg.dataset_config.n_train_samples = args.num
    pmtc_list = pre_main(ename, cfg, args, 'pretrain')

    args.atk_epochs = 30
    logger.info('attacking pretraining model!!!')
    amtc_list = atk_main(ename, cfg, args, 'pretrain')

    logger.handlers.clear()
    logging.shutdown()
    print('logging shut down!')





