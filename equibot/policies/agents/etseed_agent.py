import numpy as np
import torch
from torch import nn

from equibot.policies.utils.norm import Normalizer
from equibot.policies.utils.misc import to_torch
from equibot.policies.agents.etseed_policy import ETSEEDPolicy
from equibot.policies.utils.diffusion.lr_scheduler import get_scheduler

class ETSEEDAgent():
    

    def load_snapshot(self, save_path):
        state_dict = torch.load(save_path)
        # if hasattr(self, "encoder_handle"):
        #     del self.encoder_handle
        #     del self.noise_pred_net_handle
        self.actor.load_state_dict(self._fix_state_dict_keys(state_dict["actor"]))
