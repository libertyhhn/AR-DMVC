import logging

import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
import atk_models


# custom weights initialization called on netG and netD
import tools


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 1)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class GAN_Attack:
    def __init__(self,
                 args,
                 tar_model,
                 perb_eps = 0.3,  # fashion:0.15 mnist:0.3 patchedmnist:0.3
                 box_min = 0,
                 box_max = 1,
                 plot_save_dir = None):  # 
        self.args = args
        self.device = args.device
        self.tar_model = tar_model
        self.perb_eps = perb_eps
        self.box_min = box_min
        self.box_max = box_max
        self.batch_no = 0
        self.epoch = 0
        self.dataset = args.data_name
        self.queries = 0


        self.is_img = True
        self.plot_save_dir = plot_save_dir

        self.netGs = getattr(atk_models, args.data_name + '_gen')().to(args.device)
        self.clamp = atk_models.ClippingLayer(min_value=-perb_eps, max_value=perb_eps)  # 
        self.netDs = getattr(atk_models, args.data_name + '_dis')().to(args.device)

        # initialize all weights
        self.netGs.apply(weights_init)
        self.netDs.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netGs.parameters(),
                                            lr=0.001)
        self.optimizer_D = torch.optim.Adam(self.netDs.parameters(),
                                            lr=0.001)

        # if not os.path.exists(models_path):
        #     os.makedirs(models_path)

    def _loss_hid(self, X, adv_X):

        X = [x.to(self.device) for x in X]
        adv_X = [x.to(self.device) for x in adv_X]

        # P1 = self.tar_model(X)
        # P2 = self.tar_model(adv_X)

        self.tar_model(X)
        P1 = self.tar_model.hidden
        # Z_list = self
        self.tar_model(adv_X)
        P2 = self.tar_model.hidden

        # P2 = self.tar_model(adv_X)
        # P1 = torch.zeros_like(P2).to(P2.device)
        # P1[:, 1] = 1.0
        self.queries += 1

        delta_P = P1 - P2
        dist_loss = 0.5 * torch.sum(torch.norm(delta_P, 2, dim=1))

        return dist_loss  # is beta * ||f(x) - Cluster||^2_2

    def _loss_multi_hid_mvae(self, X, adv_X):

        X = [x.to(self.device) for x in X]
        adv_X = [adv_x.to(self.device) for adv_x in adv_X]

        # P1 = self.tar_model(X)
        # P2 = self.tar_model(adv_X)

        self.tar_model(X)
        P1 = self.tar_model.alpha
        Z1_list = self.tar_model.encoder_outputs
        self.tar_model(adv_X)
        P2 = self.tar_model.alpha
        Z2_list = self.tar_model.encoder_outputs

        # P2 = self.tar_model(adv_X)
        # P1 = torch.zeros_like(P2).to(P2.device)
        # P1[:, 1] = 1.0
        self.queries += 1

        delta_P = P1 - P2
        distP_loss = 0.5 * torch.sum(torch.norm(delta_P, 2, dim=1))

        delta_Z_list = [Z1 - Z2 for Z1, Z2 in zip(Z1_list, Z2_list)]
        delta_Z_list = [0.5 * torch.sum(torch.norm(delta_Z, 2, dim=1)) for delta_Z in delta_Z_list]
        delta_Z_loss = sum(delta_Z_list)

        dist_loss = distP_loss + delta_Z_loss
        return dist_loss  # is beta * ||f(x) - Cluster||^2_2

    def _loss_multi_hid(self, X, adv_X):

        X = [x.to(self.device) for x in X]
        adv_X = [adv_x.to(self.device) for adv_x in adv_X]

        # P1 = self.tar_model(X)
        # P2 = self.tar_model(adv_X)

        self.tar_model(X)
        P1 = self.tar_model.hidden
        Z1_list = self.tar_model.encoder_outputs
        self.tar_model(adv_X)
        P2 = self.tar_model.hidden
        Z2_list = self.tar_model.encoder_outputs

        # P2 = self.tar_model(adv_X)
        # P1 = torch.zeros_like(P2).to(P2.device)
        # P1[:, 1] = 1.0
        self.queries += 1

        delta_P = P1 - P2
        distP_loss = 0.5 * torch.sum(torch.norm(delta_P, 2, dim=1))  #

        delta_Z_list = [Z1 - Z2 for Z1, Z2 in zip(Z1_list, Z2_list)]
        delta_Z_list = [0.5 * torch.sum(torch.norm(delta_Z, 2, dim=1)) for delta_Z in delta_Z_list]
        delta_Z_loss = sum(delta_Z_list)

        dist_loss = distP_loss + delta_Z_loss
        return dist_loss  # is beta * ||f(x) - Cluster||^2_2

    def train_batch(self, views):


        perturbs = self.netGs(views)  # list
        # add a clipping trick
        perturbs = [self.clamp(item) for item in perturbs]  # 

        for i in range(1):
            adv_views = []
            for i, x in enumerate(views):
                adv_image = perturbs[i] + x
                if self.is_img:
                    adv_image = torch.clamp(adv_image, self.box_min, self.box_max)  # 
                adv_views.append(adv_image)

            self.optimizer_D.zero_grad()
            pred_real = torch.cat(self.netDs(views), dim=0)  # 
            loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
            loss_D_real.backward()

            adv_views_detach = [item.detach() for item in adv_views]  # 
            pred_fake = torch.cat(self.netDs(adv_views_detach), dim=0)  # 
            loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
            loss_D_fake.backward()
            loss_D_GAN = loss_D_fake + loss_D_real
            self.optimizer_D.step()

        # optimize G
        for i in range(1):
            self.optimizer_G.zero_grad()

            # cal G's loss in GAN
            pred_fake = torch.cat(self.netDs(adv_views), dim=0)  # 
            loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
            loss_G_fake.backward(retain_graph=True)

            # calculate perturbation norm
            loss_perturb = 0
            for p2 in perturbs:
                loss_perturb += torch.mean(torch.norm(p2.view(p2.shape[0], -1), 2, dim=1))  # 
                # loss_p = torch.mean(torch.norm(p2.view(p2.shape[0], -1), 2, dim=1))  # 
                # loss_perturb += torch.max(loss_p-self.perb_eps, torch.zeros(1, device=self.device))
            if self.args.atk_mode == 0:
                loss_adv = self._loss_hid(views, adv_views)  # works
            elif self.args.atk_mode == 2:
                loss_adv = self._loss_multi_hid_mvae(views, adv_views)  # works
            else:
                loss_adv = self._loss_multi_hid(views, adv_views)
            loss_adv = -loss_adv  # works

            adv_lambda = 5  # 5
            pert_lambda = 1
            loss_G = pert_lambda * loss_perturb + adv_lambda * loss_adv
            # loss_G = loss_adv
            loss_G.backward()
            self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G.item(), loss_perturb.item(), loss_adv.item(), adv_views_detach
        # return 0, loss_G.item(), loss_perturb.item(), loss_adv.item(), 0

    def train(self, train_dataloader, test_dataloader, epochs, atk_save_dir=None):

        mtc_list = []
        mtc = self.val_real(test_dataloader)
        mtc_list.append(np.array([mtc['acc'], mtc['nmi'], mtc['ari']]))
        for epoch in range(1, epochs + 1):
            self.epoch = epoch
            if epoch == 20:
                self.optimizer_G = torch.optim.Adam(self.netGs.parameters(),
                                                    lr=0.0001)
                self.optimizer_D = torch.optim.Adam(self.netDs.parameters(),
                                                    lr=0.0001)

            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0

            torch.cuda.empty_cache()

            self.netGs.train()
            self.netDs.train()
            for batch_idx, Data in enumerate(train_dataloader):
                views = Data[0:-1]
                labels = Data[-1]

                views = [item.to(self.device) for item in views]
                labels.to(self.device)
                # train
                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch, adv_views_detach = \
                    self.train_batch(views)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            # 
            mtc = self.val_fake(test_dataloader)
            mtc_list.append(np.array([mtc['acc'], mtc['nmi'], mtc['ari']]))

            # print statistics
            self.batch_no += 1
            num_batch = len(train_dataloader)
            logging.info("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum / num_batch, loss_G_fake_sum / num_batch,
                   loss_perturb_sum / num_batch, loss_adv_sum / num_batch))

            if atk_save_dir is not None:
                torch.save(self.netGs.state_dict(), str(atk_save_dir) + '/{}.ckpt'.format(epoch))
                logging.info('atk model has been saved in {}/{}.ckpt'.format(str(atk_save_dir), epoch))
        return mtc, mtc_list

    def plot_images(self, views, adv_views, mtc):
        imgs_tensor_list = []
        title_list = []
        for i in range(len(views)):
            imgs_tensor_list.append(views[i][:5])
            title_list.append('v{}_real'.format(i+1))
            imgs_tensor_list.append(adv_views[i][:5])
            title_list.append('v{}_fake'.format(i + 1))
        args = {
            'epoch': self.epoch,
            'perb_eps': self.perb_eps,
            'acc': mtc['acc']
        }
        # if self.fig_saved_dir is not None:
        #     fig_saved_path
        # hang_tools.plot_result(imgs_tensor_list, title_list, args, fig_saved_path=self.fig_saved_dir)

    def plot_ad_images(self, test_loader, fig_saved_dir=None):


        data_base = next(iter(test_loader))  # 
        # origin_base
        views_base = data_base[:-2]
        views_base = [item.to(self.device, non_blocking=True) for item in views_base]
        label_base = data_base[-2]
        plabel_base = data_base[-1]

        # perb_base
        perturbs_base = self.netGs(views_base)  # list
        perturbs_base = [self.clamp(item) for item in perturbs_base]
        # perturbs_base_ = [torch.clamp(item, 0, 1) for item in perturbs_base]  # 负扰动置0
        # ad_base
        adv_views_base = []
        for i, x in enumerate(views_base):
            adv_image = perturbs_base[i] + x
            if self.is_img:
                adv_image = torch.clamp(adv_image.detach(), self.box_min, self.box_max)
            adv_views_base.append(adv_image)

        # prepare data
        view_tensor_lists = [[] for v in range(len(views_base))]
        # view2_tensor_list = []
        label_list = []
        plabel_list = []
        for i in range(len(label_base)):

            if label_base[i] == plabel_base[i]:
                continue
            label_list.append(int(label_base[i]))
            plabel_list.append(int(plabel_base[i]))

            for v in range(len(views_base)):
                view_tensor_lists[v].append(torch.stack([views_base[v][i], perturbs_base[v][i], adv_views_base[v][i]], dim=0))

            if len(label_list) == 10:
                break

        for v in range(len(views_base)):
            if fig_saved_dir is not None:
                fig_saved_path = str(fig_saved_dir) + '/view{}.eps'.format(v)
            else:
                fig_saved_path = None
            tools.plot_ad_result(view_tensor_lists[v], label_list, plabel_list, fig_saved_path)

    def plot_ad_images_kmeans(self, test_loader, kmeans_to_true_cluster_labels=None):

        data_base = next(iter(test_loader))  # 
        # origin_base
        views_base = data_base[:-1]
        views_base = [item.to(self.device, non_blocking=True) for item in views_base]
        label_base = data_base[-1]
        print(label_base)
        # perb_base
        perturbs_base = self.netGs(views_base)  # list
        perturbs_base = [self.clamp(item) for item in perturbs_base]
        # perturbs_base_ = [torch.clamp(item, 0, 1) for item in perturbs_base]  # 负扰动置0
        # ad_base
        adv_views_base = []
        for i, x in enumerate(views_base):
            adv_image = perturbs_base[i] + x
            if self.is_img:
                adv_image = torch.clamp(adv_image.detach(), self.box_min, self.box_max)
            adv_views_base.append(adv_image)
        # plabel_base
        pred_base = list(range(len(adv_views_base[0])))
        print(pred_base)
        pred_a_base = list(range(len(adv_views_base[0])))
        print(pred_a_base)

        # prepare data
        view1_tensor_list = []
        view2_tensor_list = []
        label_list = []
        plabel_list = []
        for i in range(len(label_base)):
            # if (pred_a_base[i] == label_base[i]) or (label_base[i] in label_list):
            #     continue
            label_list.append(int(label_base[i]))
            plabel_list.append(int(pred_a_base[i]))
            # if self.args.data_name == 'regdb':
            #     from hang_tools import clf_to_raw
            #     retransform = clf_to_raw()
            #     for v in range(2):
            #         views_base[v][i] = torch.clamp(retransform(views_base[v][i]), self.box_min, self.box_max)
            #         adv_views_base[v][i] = torch.clamp(retransform(adv_views_base[v][i]), self.box_min, self.box_max)
            view1_tensor_list.append(torch.stack([views_base[0][i], perturbs_base[0][i], adv_views_base[0][i]], dim=0))
            view2_tensor_list.append(torch.stack([views_base[1][i], perturbs_base[1][i], adv_views_base[1][i]], dim=0))

            if len(label_list) == 10:
                break

        tools.plot_ad_result(view1_tensor_list, label_list, plabel_list)
        tools.plot_ad_result(view2_tensor_list, label_list, plabel_list)

    def val_fake(self, test_loader, is_get_pred=False):
        adv_list = []
        self.netGs.eval()
        save_images = None
        for idx, batch in enumerate(test_loader):
            batch = [item.to(self.device, non_blocking=True) for item in batch]
            views = batch[0:-1]
            label = batch[-1]

            perturbs = self.netGs(views)  # list
            perturbs = [self.clamp(item) for item in perturbs]
            adv_views = []
            for i, x in enumerate(views):
                adv_image = perturbs[i] + x
                if self.is_img:
                    adv_image = torch.clamp(adv_image, self.box_min, self.box_max)
                adv_views.append(adv_image.detach())

            # 
            if idx == 0 and self.epoch % 1 == 0:
                if self.is_img:  # 
                    save_images = [views, adv_views]

            adv_views.append(label)
            adv_list.append(self.tar_model._val_test_step(adv_views, idx, 'test'))

        if is_get_pred:
            mtc, pred = self.tar_model._val_test_epoch_end(adv_list, 'test', is_get_pred=is_get_pred)
            logging.info('fake data metric: acc:{:4f} nmi:{:4f} ari:{:4f}'.format(mtc['acc'], mtc['nmi'], mtc['ari']))

            return mtc, pred
        else:
            mtc = self.tar_model._val_test_epoch_end(adv_list, 'test')
            logging.info(
                'fake data metric: acc:{:4f} nmi:{:4f} ari:{:4f}'.format(mtc['acc'], mtc['nmi'], mtc['ari']))
            if save_images is not None:
                self.plot_images(save_images[0], save_images[1], mtc)

            return mtc

    def val_real(self, test_loader, is_get_pred=False):
        test_list = []
        for idx, batch in enumerate(test_loader):
            batch = [item.to(self.device) for item in batch]
            test_list.append(self.tar_model._val_test_step(batch, idx, 'test'))


        if is_get_pred:
            mtc, pred = self.tar_model._val_test_epoch_end(test_list, 'test', is_get_pred=is_get_pred)
            logging.info('real data metric: acc:{:4f} nmi:{:4f} ari:{:4f}'.format(mtc['acc'], mtc['nmi'], mtc['ari']))

            return mtc, pred
        else:
            mtc = self.tar_model._val_test_epoch_end(test_list, 'test')
            logging.info(
                'real data metric:acc:{:4f} nmi:{:4f} ari:{:4f}'.format(mtc['acc'], mtc['nmi'], mtc['ari']))
            return mtc


    def get_kmeans_to_true_cluster_labels(self, test_loader):

        import lib.metrics as metrics
        test_list = []
        for idx, batch in enumerate(test_loader):
            batch = [item.to(self.device, non_blocking=True) for item in batch]
            test_list.append(self.tar_model._val_test_step(batch, idx, 'test'))

        labels = np.concatenate(test_list, axis=1)
        t_labels = labels[0]
        p_labels = labels[1]

        n_clusters = np.size(np.unique(t_labels))

        confusion_matrix = metrics.confusion_matrix(t_labels, p_labels, labels=None)  # 
        # compute accuracy based on optimal 1:1 assignment of clusters to labels
        cost_matrix = metrics.calculate_cost_matrix(confusion_matrix, n_clusters)  # 
        indices = metrics.Munkres().compute(cost_matrix)
        kmeans_to_true_cluster_labels = metrics.get_cluster_labels_from_indices(indices)

        print('got kmeans_to_true_cluster_labels!!!')
        print('true labels:{}'.format(t_labels[:30]))
        print('predict labels:{}'.format(p_labels[:30]))
        print('predict to true labels:{}'.format(kmeans_to_true_cluster_labels[p_labels[:30]]))


        return kmeans_to_true_cluster_labels



