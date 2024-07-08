import torch as th
from torch import nn

from lib.fusion import get_fusion_module
from lib.encoder import EncoderList, Encoder
from lib.normalization import get_normalizer
from models.clustering_module import get_clustering_module
from models.base import BaseModel
from register import register_model


@register_model
class GEN(BaseModel):
    def __init__(self, cfg):
        super(GEN, self).__init__(cfg)

        self.decoders = EncoderList(cfg.decoder_configs, input_sizes=self.encoders.output_sizes_before_flatten)

        self.views = None
        self.encoder_outputs = None
        self.decoder_outputs = None


    def forward(self, views):
        self.views = views
        self.encoder_outputs = self.encoders(views)

        self.decoder_outputs = self.decoders(
            [inp.view(-1, *size) for inp, size in zip(self.encoder_outputs, self.encoders.output_sizes_before_flatten)]
        )

        return self.decoder_outputs

@register_model
class DIS(BaseModel):
    def __init__(self, cfg):
        super(DIS, self).__init__(cfg)

        self.views = None
        self.projector = EncoderList(encoder_modules=[
            nn.Sequential(
                nn.Linear(in_features=self.encoders.output_sizes[v][0], out_features=100),
                nn.ReLU(),
                nn.Linear(in_features=100, out_features=1),
                nn.Sigmoid()
            ) for v in range(self.n_views)])


    def forward(self, views):
        self.views = views
        self.encoder_outputs = self.encoders(views)

        self.projections = [self.projector[v](x) for v, x in enumerate(self.encoder_outputs)]
        self.projections = [x.squeeze() for x in self.projections]

        return self.projections

