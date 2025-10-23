# coding : utf-8
# Author : Yuxiang Zeng
# 每次开展新实验都改一下这里
from models.layers.metric.distance import PairwiseLoss
from exp.exp_base import BasicModel
from models.backbone import Backbone


class Model(BasicModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.input_size = config.input_size
        self.hidden_size = config.d_model
        if config.model == 'ours':
            self.model = Backbone(self.input_size, config)

        else:
            raise ValueError(f"Unsupported model type: {config.model}")


        