import logging
import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.cluster import MiniBatchKMeans



def mini_kmeans(x_all, class_num, batch_size=3000):
    kmeans_assignments = []
    kmeans = MiniBatchKMeans(n_clusters=class_num, batch_size=batch_size)
    for i in range(0, len(x_all), batch_size):
        kmeans.partial_fit(x_all[i:i + batch_size])
    for i in range(0, len(x_all), batch_size):
        k_res = kmeans.predict(x_all[i:i + batch_size])
        kmeans_assignments.append(k_res)
    kmeans_assignments = np.concatenate(kmeans_assignments)

    return kmeans_assignments  # y_pred

def plot_ad_result(imgdata, label_list, plabel_list, fig_saved_path=None, use_clip=True, cmap=None):

    dtype = len(imgdata)
    num = len(imgdata[0])
    fig, axes = plt.subplots(dtype, num, figsize=(dtype * num, 5*dtype))
    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
    for d in range(dtype):

        if cmap is None:
            cmap = 'viridis' if imgdata[d].shape[1] == 3 else 'gray'
        temp_data = imgdata[d].permute(0, 2, 3, 1).cpu().detach().numpy()

        # temp_data = (temp_data * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
        temp_data= temp_data * 255
        if use_clip:
            temp_data = np.clip(temp_data, 0, 255)
        temp_data = temp_data.astype('uint8')

        for n in range(num):
            im_result = temp_data[n]
            axes[d][n].set_xticks([])
            axes[d][n].set_yticks([])
            if n == 0:
                axes[d][n].set_title(label_list[d], fontsize=45)
            elif n == 2:
                axes[d][n].set_title(plabel_list[d], fontsize=45)

            axes[d][n].imshow(im_result, cmap=cmap)

    if fig_saved_path is not None:
        fig.savefig(fig_saved_path)
        print('fig have saved!!! path is {}'.format(fig_saved_path))
    fig.show()


def log(log_dirname, log_filename='', exp_id='', is_cover=None, is_iterate=None):
    project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), ''))
    # project_path = os.path.dirname(os.path.realpath(__file__))
    log_path = project_path + '/' + log_dirname
    if not os.path.exists(log_path):
        os.makedirs(log_path)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    log_save = log_path + '/' + log_filename + '_' + exp_id + '.log'
    if is_cover:
        if os.path.exists(log_save):
            os.remove(log_save)
    if is_iterate:
        while 1:
            if os.path.exists(log_save):
                exp_id += 1
                log_save = log_path + '/' + log_filename + '_' + str(exp_id) + '.log'
                continue
            fh = logging.FileHandler(log_save, encoding='utf8')
            fh.setFormatter(formatter)
            logger.addHandler(fh)
            return logger, exp_id

    fh = logging.FileHandler(log_save, encoding='utf8')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger
