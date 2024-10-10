import numpy as np
import torch as th
import pytorch_lightning as pl
from sklearn.cluster import MiniBatchKMeans, KMeans

import helpers
from lib.metrics import calc_metrics
from lib.encoder import EncoderList
from lib.loss import Loss


class BaseModel(pl.LightningModule):
    def __init__(self, cfg, flatten_encoder_output=True):
        super(BaseModel, self).__init__()
        self.cfg = cfg
        self.n_views = cfg.n_views
        self.data_module = None

        self.encoders = EncoderList(cfg.encoder_configs, flatten_output=flatten_encoder_output)

        self.loss = None
        self.init_losses()

    def init_losses(self):
        self.loss = Loss(self.cfg.loss_config, self)

    @property
    def requires_pre_train(self):
        return False

    def attach_data_module(self, data_module):
        self.data_module = data_module

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def get_loss(self):
        return self.loss(self)

    @staticmethod
    def _optimizer_from_cfg(cfg, params):
        if cfg.opt_type == "adam":
            optimizer = th.optim.Adam(params, lr=cfg.learning_rate)
        elif cfg.opt_type == "sgd":
            optimizer = th.optim.SGD(params, lr=cfg.learning_rate, momentum=cfg.sgd_momentum)
        else:
            raise RuntimeError()

        if getattr(cfg, "scheduler_config", None) is None:
            # We didn't get a scheduler-config
            return optimizer

        s_cfg = cfg.scheduler_config

        if s_cfg.warmup_epochs is not None:
            if s_cfg.warmup_epochs > 0:
                # Linear warmup scheduler
                warmup_lrs = np.linspace(0, 1, s_cfg.warmup_epochs + 1, endpoint=False)[1:]
                scheduler_lambda = lambda epoch: warmup_lrs[epoch] if epoch < s_cfg.warmup_epochs else 1.0
            else:
                scheduler_lambda = lambda epoch: 1.0
            scheduler = th.optim.lr_scheduler.LambdaLR(optimizer, scheduler_lambda)
        else:
            # Multiplicative decay scheduler
            assert (s_cfg.step_size is not None) and (s_cfg.gamma is not None)
            scheduler = th.optim.lr_scheduler.StepLR(optimizer, step_size=s_cfg.step_size, gamma=s_cfg.gamma)

        scheduler = {
            "scheduler": scheduler,
            "name": "learning_rate",
        }

        return [optimizer], [scheduler]

    def configure_optimizers(self):
        return self._optimizer_from_cfg(self.cfg.optimizer_config, self.parameters())

    def _log_dict(self, dct, prefix, sep="/", ignore_keys=tuple()):
        if prefix:
            prefix += sep
        for key, value in dct.items():
            if key not in ignore_keys:
                self.log(prefix + key, float(value))

    def split_batch(self, batch, **_):
        assert len(batch) == (self.n_views + 1), f"Invalid number of tensors in batch ({len(batch)}) for model " \
                                                   f"{self.__class__.__name__}"
        views = batch[:self.n_views]
        labels = batch[-1]
        return views, labels

    def _train_step(self, batch):
        *inputs, labels = self.split_batch(batch, includes_labels=True)
        _ = self(*inputs)
        losses = self.get_loss()

        self._log_dict(losses, prefix="train_loss")
        return losses["tot"]

    def training_step(self, batch, idx):
        return self._train_step(batch)

    def _val_test_step(self, batch, idx, prefix):
        *inputs, labels = self.split_batch(batch, includes_labels=True)
        pred = self(*inputs)  # 这里得到聚类表征

        # Only evaluate losses on full batches
        if labels.size(0) == self.cfg.batch_size:
            losses = self.get_loss()
            self._log_dict(losses, prefix=f"{prefix}_loss")

        return np.stack((helpers.npy(labels), helpers.npy(pred).argmax(axis=1)), axis=0)  # views x batch_size 累积到 step_outputs

    def _val_test_epoch_end(self, step_outputs, prefix, is_get_pred=False):  # list:[700x[2x100]]
        if not isinstance(step_outputs, list):
            step_outputs = [step_outputs]

        labels_pred = np.concatenate(step_outputs, axis=1)

        # mtc = calc_metrics(labels=labels_pred[0], pred=labels_pred[1])
        # self._log_dict(mtc, prefix=f"{prefix}_metrics")
        # return mtc
        return calc_metrics(labels=labels_pred[0], pred=labels_pred[1], is_get_pred=is_get_pred)

    def validation_step(self, batch, idx):
        return self._val_test_step(batch, idx, "val")

    def validation_epoch_end(self, step_outputs):
        return self._val_test_epoch_end(step_outputs, "val")

    # 只需要这一步
    def test_step(self, batch, idx):
        return self._val_test_step(batch, idx, self.test_prefix)

    def test_epoch_end(self, step_outputs):
        return self._val_test_epoch_end(step_outputs, self.test_prefix)

    def _val_test_step_kmeans(self, batch, idx, prefix):
        *inputs, labels = self.split_batch(batch, includes_labels=True)

        return helpers.npy(labels), helpers.npy(self.output)  # self.eval_tensors

    def _val_test_epoch_end_kmeans(self, step_outputs, prefix, is_get_pred=False):
        labels = np.concatenate([s[0] for s in step_outputs], axis=0)

        eval_tensors = np.concatenate([s[1] for s in step_outputs], axis=0)
        assert eval_tensors.ndim in {2, 3}
        if eval_tensors.ndim == 3:
            eval_tensors = np.concatenate([eval_tensors[:, v] for v in range(eval_tensors.shape[1])], axis=1)

        # FAISS-kmeans seems to be significantly worse than sklearn.
        # pred, *_ = helpers.faiss_kmeans(eval_tensors, self.cfg.n_clusters, n_iter=300, n_redo=10)
        # pred = KMeans(n_clusters=self.cfg.n_clusters, n_init=10).fit_predict(eval_tensors)  # 这里需要修改 mvae mviic

        if len(eval_tensors) > 15000:
            x_list = eval_tensors
            kmeans_assignments = []
            batch_size = 3000
            kmeans = MiniBatchKMeans(n_clusters=self.cfg.n_clusters, batch_size=batch_size)
            for i in range(0, len(x_list), batch_size):
                kmeans.partial_fit(x_list[i:i + batch_size])
            for i in range(0, len(x_list), batch_size):
                k_res = kmeans.predict(x_list[i:i + batch_size])
                kmeans_assignments.append(k_res)

            pred = np.concatenate(kmeans_assignments)
        else:
            pred = KMeans(n_clusters=self.cfg.n_clusters, n_init=10).fit_predict(eval_tensors)  # 这里需要修改 mvae mviic

        # mtc = metrics.calc_metrics(labels=labels, pred=pred)
        # self._log_dict(mtc, prefix=f"{prefix}_metrics")
        return calc_metrics(labels=labels, pred=pred, is_get_pred=is_get_pred)
