import logging
import random
import warnings

import torch
from torch.utils.data import TensorDataset

from data.data_module import DataModule
from models.SEM import Network, valid
import numpy as np
import argparse
from models.SEM import Loss
import os
import time

import config
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def ori_train(cfg, model, data_loader, test_loader, args, logger):
    accs = []
    nmis = []
    aris = []
    purs = []
    ACC_tmp = 0

    for Runs in range(1):   # 10
        print("ROUND:{}".format(Runs+1))

        t1 = time.time()
        # set_seed(5)   # if we find that the initialization of networks is sensitive, we can set a seed for stable performance.

        view = cfg.n_views
        device = args.device
        class_num = cfg.n_clusters
        data_size = len(data_loader.dataset)
        def Low_level_rec_train(epoch, rec='AE', p=0.3, mask_ones_full=[], mask_ones_not_full=[]):
            tot_loss = 0.
            criterion = torch.nn.MSELoss()
            Vones_full = []
            Vones_not_full = []
            flag_full = 0
            flag_not_full = 0
            for batch_idx, data in enumerate(data_loader):
                # if batch_idx == 10:
                #     break
                xs = data[:-1]
                xs = [xs[v].to(device) for v in range(view)]
                # for v in range(view):
                #     xs[v] = xs[v].to(device)

                xnum = xs[0].shape[0]

                if rec == 'AE':
                    optimizer.zero_grad()
                    _, _, xrs, _, _ = model(xs)
                if rec == 'DAE':
                    noise_x = []
                    for v in range(view):
                        # print(xs[v])
                        noise = torch.randn(xs[v].shape).to(device)
                        # print(noise)
                        noise = noise + xs[v]
                        # print(noise)
                        noise_x.append(noise)
                    optimizer.zero_grad()
                    _, _, xrs, _, _ = model(noise_x)
                if rec == 'MAE':
                    noise_x = []
                    for v in range(view):

                        if xnum == args.batch_size and flag_full == 0 and epoch == 1:
                            # print(1)
                            num = xs[v].shape[0] * xs[v].shape[1] * xs[v].shape[2] * xs[v].shape[3]
                            ones = torch.ones([1, num]).to(device)
                            zeros_num = int(num * p)
                            for i in range(zeros_num):
                                ones[0, i] = 0
                            Vones_full.append(ones)
                        if xnum is not args.batch_size and flag_not_full == 0 and epoch == 1:
                            # print(1)
                            num = xs[v].shape[0] * xs[v].shape[1]
                            ones = torch.ones([1, num]).to(device)
                            zeros_num = int(num * p)
                            for i in range(zeros_num):
                                ones[0, i] = 0
                            Vones_not_full.append(ones)

                        if xnum == args.batch_size and epoch == 1:
                            noise = Vones_full[v][:, torch.randperm(Vones_full[v].size(1))]
                        if xnum is not args.batch_size and epoch == 1:
                            noise = Vones_not_full[v][:, torch.randperm(Vones_not_full[v].size(1))]

                        if xnum == args.batch_size and epoch is not 1:
                            noise = mask_ones_full[v][:, torch.randperm(mask_ones_full[v].size(1))]
                        if xnum is not args.batch_size and epoch is not 1:
                            noise = mask_ones_not_full[v][:, torch.randperm(mask_ones_not_full[v].size(1))]
                        noise = torch.reshape(noise, xs[v].shape)
                        noise = noise * xs[v]
                        noise_x.append(noise)

                    if xnum == args.batch_size:
                        flag_full = 1
                    else:
                        flag_not_full = 1

                    optimizer.zero_grad()
                    _, _, xrs, _, _ = model(noise_x)

                loss_list = []
                for v in range(view):
                    loss_list.append(criterion(xs[v], xrs[v]))
                loss = sum(loss_list)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()
            # print('Epoch {}'.format(epoch), 'Loss:{:.6f}'.format(tot_loss / len(data_loader)))
            return Vones_full, Vones_not_full

        def High_level_contrastive_train(epoch, nmi_matrix, Lambda=1.0, rec='AE', p=0.3, mask_ones_full=[], mask_ones_not_full=[]):
            tot_loss = 0.
            mes = torch.nn.MSELoss()
            record_loss_con = []
            Vones_full = []
            Vones_not_full = []
            flag_full = 0
            flag_not_full = 0

            for v in range(view):
                record_loss_con.append([])
                for w in range(view):
                    record_loss_con[v].append([])

            # Sim = 0
            # cos = torch.nn.CosineSimilarity(dim=0)

            for batch_idx, data in enumerate(data_loader):
                # if batch_idx == 10:
                #     break
                xs = data[:-1]
                xs = [xs[v].to(device) for v in range(view)]
                # for v in range(view):
                #     xs[v] = xs[v].to(device)

                optimizer.zero_grad()
                zs, qs, xrs, hs, re_h = model(xs)
                loss_list = []

                xnum = xs[0].shape[0]
                #------------------------
                # P = zs[0]
                # Q = zs[1]
                # for i in range(xnum):
                #     # print(cos(P[i], Q[i]))
                #     Sim += cos(P[i], Q[i]).item()
                #-------------------------
                if rec == 'DAE':
                    noise_x = []
                    for v in range(view):
                        # print(xs[v])
                        noise = torch.randn(xs[v].shape).to(device)
                        # print(noise)
                        noise = noise + xs[v]
                        # print(noise)
                        noise_x.append(noise)
                    optimizer.zero_grad()
                    _, _, xrs, _, _ = model(noise_x)
                if rec == 'MAE':
                    noise_x = []
                    for v in range(view):

                        if xnum == args.batch_size and flag_full == 0 and epoch == 1:
                            # print(1)
                            # num = xs[v].shape[0] * xs[v].shape[1]
                            num = xs[v].shape[0] * xs[v].shape[1] * xs[v].shape[2] * xs[v].shape[3]
                            ones = torch.ones([1, num]).to(device)
                            zeros_num = int(num * p)
                            for i in range(zeros_num):
                                ones[0, i] = 0
                            Vones_full.append(ones)
                        if xnum is not args.batch_size and flag_not_full == 0 and epoch == 1:
                            # print(1)
                            num = xs[v].shape[0] * xs[v].shape[1]
                            ones = torch.ones([1, num]).to(device)
                            zeros_num = int(num * p)
                            for i in range(zeros_num):
                                ones[0, i] = 0
                            Vones_not_full.append(ones)

                        if xnum == args.batch_size and epoch == 1:
                            noise = Vones_full[v][:, torch.randperm(Vones_full[v].size(1))]
                        if xnum is not args.batch_size and epoch == 1:
                            noise = Vones_not_full[v][:, torch.randperm(Vones_not_full[v].size(1))]

                        if xnum == args.batch_size and epoch is not 1:
                            noise = mask_ones_full[v][:, torch.randperm(mask_ones_full[v].size(1))]
                        if xnum is not args.batch_size and epoch is not 1:
                            noise = mask_ones_not_full[v][:, torch.randperm(mask_ones_not_full[v].size(1))]

                        noise = torch.reshape(noise, xs[v].shape)
                        noise = noise * xs[v]
                        noise_x.append(noise)

                    if xnum == args.batch_size:
                        flag_full = 1
                    else:
                        flag_not_full = 1

                    optimizer.zero_grad()
                    _, _, xrs, _, _ = model(noise_x)

                for v in range(view):
                    # for w in range(v + 1, view):
                    for w in range(view):
                        # if v == w:
                        #     continue
                        if args.contrastive_loss == 'InfoNCE':
                            tmp = criterion.forward_feature_InfoNCE(zs[v], zs[w], batch_size=xnum)
                        if args.contrastive_loss == 'PSCL':
                            tmp = criterion.forward_feature_PSCL(zs[v], zs[w])
                        if args.contrastive_loss == 'RINCE':
                            tmp = criterion.forward_feature_RINCE(zs[v], zs[w], batch_size=xnum)

                        # loss_list.append(tmp)
                        loss_list.append(tmp * nmi_matrix[v][w])
                        record_loss_con[v][w].append(tmp)

                    loss_list.append(Lambda * mes(xs[v], xrs[v]))
                loss = sum(loss_list)
                loss.backward()
                optimizer.step()
                tot_loss += loss.item()

            # print(Sim / 1400)  # 1400 is the data size of Caltech

            for v in range(view):
                for w in range(view):
                    record_loss_con[v][w] = sum(record_loss_con[v][w])
                    record_loss_con[v][w] = record_loss_con[v][w].item() / len(data_loader)

            return Vones_full, Vones_not_full, record_loss_con, None

        # if not os.path.exists('./models'):
        #     os.makedirs('./models')

        # model = Network(view, dims, args.feature_dim, args.high_feature_dim, class_num, device)
        # print(model)
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        criterion = Loss(args.batch_size, class_num, args.temperature_f, device).to(device)

        print("Initialization......")
        epoch = 0
        while epoch < args.mse_epochs:
            epoch += 1
            if epoch == 1:
                mask_ones_full, mask_ones_not_full = Low_level_rec_train(epoch,
                                                                         rec=args.Recon,
                                                                         p=per,
                                                                         )
            else:
                Low_level_rec_train(epoch,
                                    rec=args.Recon,
                                    p=per,
                                    mask_ones_full=mask_ones_full,
                                    mask_ones_not_full=mask_ones_not_full,
                                    )
            # logger.info('pretrain ae epoch:{}'.format(epoch))
            # valid(model, device, test_loader, view, data_size, class_num,
            #     eval_h=True, eval_z=False, times_for_K=args.times_for_K,
            #     Measure=args.measurement, test=False, sample_num=sample_mmd,
            #     logger=logger)
        acc, nmi, ari, pur, nmi_matrix_1, _ = valid(model, device, test_loader, view, data_size, class_num,
                                                    eval_h=True, eval_z=False, times_for_K=args.times_for_K,
                                                    Measure=args.measurement, test=False, sample_num=sample_mmd, logger=logger)

        print("Self-Weighted Multi-view Contrastive Learning with Reconstruction Regularization...")
        Iteration = 1
        print("Iteration " + str(Iteration) + ":")
        epoch = 0
        record_loss_con = []
        record_cos = []
        while epoch < args.Total_con_epochs:
            epoch += 1
            if epoch == 1:
                mask_ones_full, mask_ones_not_full, record_loss_con_, record_cos_ = High_level_contrastive_train(epoch,
                                                                                                 nmi_matrix_1,
                                                                                                 args.Lambda,
                                                                                                 rec=args.Recon,
                                                                                                 p=per)
            else:
                _, _, record_loss_con_, record_cos_ = High_level_contrastive_train(epoch,
                                                                      nmi_matrix_1,
                                                                      args.Lambda,
                                                                      rec=args.Recon,
                                                                      p=per,
                                                                      mask_ones_full=mask_ones_full,
                                                                      mask_ones_not_full=mask_ones_not_full,
                                                                      )

            record_loss_con.append(record_loss_con_)
            record_cos.append(record_cos_)

            # logger.info('train epoch:{}'.format(epoch))
            # valid(model, device, test_loader, view, data_size, class_num,
            #                                             eval_h=False, eval_z=True, times_for_K=args.times_for_K,
            #                                             Measure=args.measurement, test=False, sample_num=sample_mmd,
            #                                             logger=logger)

            if epoch % args.con_epochs == 0:
                if epoch == args.mse_epochs + args.Total_con_epochs:
                    break

                # print(nmi_matrix_1)

                acc, nmi, ari, pur, _, nmi_matrix_2 = valid(model, device, test_loader, view, data_size, class_num,
                                                            eval_h=False, eval_z=True, times_for_K=args.times_for_K,
                                                            Measure=args.measurement, test=False, sample_num=sample_mmd, logger=logger)
                nmi_matrix_1 = nmi_matrix_2
                if epoch < args.Total_con_epochs:
                    Iteration += 1
                    print("Iteration " + str(Iteration) + ":")

            pg = [p for p in model.parameters() if p.requires_grad]
            #  this code matters, to re-initialize the optimizers
            optimizer = torch.optim.Adam(pg, lr=args.learning_rate, weight_decay=args.weight_decay)

        accs.append(acc)
        nmis.append(nmi)
        aris.append(ari)
        purs.append(pur)

        # if acc > ACC_tmp:
        #     ACC_tmp = acc
        #     state = model.state_dict()
        #     torch.save(state, './models/' + args.dataset + '.pth')

        t2 = time.time()
        print("Time cost: " + str(t2 - t1))
        print('End......')


    print(accs, np.mean(accs)/0.01, np.std(accs)/0.01)
    print(nmis, np.mean(nmis)/0.01, np.std(nmis)/0.01)
    # print(aris, np.mean(aris)/0.01, np.std(aris)/0.01)
    # print(purs, np.mean(purs)/0.01, np.std(purs)/0.01)
CL_Loss = ['InfoNCE', 'PSCL', 'RINCE']  # three kinds of contrastive losses
Measure_M_N = ['CMI', 'JSD',
               'MMD']  # Class Mutual Information (CMI), Jensen–Shannon Divergence (JSD), Maximum Mean Discrepancy (MMD)
sample_mmd = 2000  # select partial samples to compute MMD as it has high complexity, otherwise might be out-of-memory
Reconstruction = ['AE', 'DAE', 'MAE']  # autoencoder (AE), denoising autoencoder (DAE), masked autoencoder (MAE)
per = 0.3  # the ratio of masked samples to perform masked AE, e.g., 30%

def sem_train(data_name, net, data_loader, test_loader, logger):

    class MyNamespace:
        def __init__(self):
            self.batch_size = 256
            self.temperature_f = 1.0
            self.contrastive_loss = CL_Loss[0]
            self.measurement = Measure_M_N[0]
            self.Recon = Reconstruction[0]
            self.bi_level_iteration = 4
            self.times_for_K = 1
            self.Lambda = 1
            self.learning_rate = 0.0003
            self.weight_decay = 0.
            self.workers = 8
            self.mse_epochs = 100
            self.con_epochs = 100
            self.feature_dim = 512
            self.high_feature_dim = 128

    args = MyNamespace()
    args.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print('SEM + ' + args.contrastive_loss + ' + ' + args.measurement + ' + ' + args.Recon)


    if data_name in ['noisymnist', 'noisyfashionmnist']:
        args.num = 70000
        args.view = 2
        args.class_num = 10
        args.dims = [28 * 28, 28 * 28]
        args.atk_eps = 0.3 if data_name == 'noisymnist' else 0.15
    elif data_name == 'patchedmnist':
        args.num = 21770
        args.view = 3
        args.class_num = 3
        args.dims = [28 * 28, 28 * 28, 28*28]
        args.mse_epochs = 30
        args.con_epochs = 30
        args.Recon = Reconstruction[1]
        args.contrastive_loss = CL_Loss[0]
        args.measurement = Measure_M_N[2]
        args.atk_eps = 0.3

    args.bi_level_iteration = 3
    args.Total_con_epochs = args.con_epochs * args.bi_level_iteration

    args.percent = 0.5


    logger.info('pretraining origin model!!!')

    _, cfg = config.get_experiment_config(data_name, 'comvc')  # 获取训练集参数

    cfg.dataset_config.n_train_samples = args.num
    cfg.dataset_config.batch_size = args.batch_size

    ename = data_name + '_' + 'SEM'

    set_seed(5)
    n = int(args.percent * args.num)
    data_module = DataModule(cfg.dataset_config)  # 在此处完成shuffle
    data_module.train_dataset = TensorDataset(*(data_module.train_dataset[:n]))

    data_loader = data_module.train_dataloader(shuffle=True, drop_last=True)
    test_loader = data_module.train_dataloader(shuffle=True, drop_last=False)

    args.data_size = len(data_module.train_dataset)
    logger.info('{} used {} samples'.format(ename, args.data_size))

    ori_train(cfg, net, data_loader, test_loader, args, logger)


