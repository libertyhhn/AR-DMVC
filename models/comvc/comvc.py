import numpy as np
from sklearn.cluster import MiniBatchKMeans, KMeans
from torch import nn
import torch as th

import helpers
from lib.metrics import calc_metrics
from models.simvc.simvc import SiMVC
from lib import encoder, metrics
from register import register_model


@register_model
class CoMVC(SiMVC):
    def __init__(self, cfg):
        super(CoMVC, self).__init__(cfg)

        if cfg.projector_config is None:
            self.projector = nn.Identity()
        else:
            self.projector = encoder.Encoder(cfg.projector_config)

        self.projections = None

    def forward(self, views):
        self.encoder_outputs = self.encoders(views)
        self.projections = [self.projector(x) for x in self.encoder_outputs]
        self.fused = self.fusion(self.encoder_outputs)
        self.hidden, self.output = self.clustering_module(self.fused)
        return self.output

    @property
    def eval_tensors(self):
        # Override to run k-means on different tensor.
        return th.stack(self.encoder_outputs, dim=1).detach()

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
