import torch
import utils
import losses
import networks
import itertools
from collections import OrderedDict
from base_model import Base

class W1(Base):
    """Wasserstein-1 based models including WGAN-GP/WGAN-LP"""

    def get_data(self, config):
        """override z with gz"""
        z = utils.to_var(next(self.z_generator)[0])
        gz = self.g(z)
        r = utils.to_var(next(self.r_generator)[0])
        if gz.size() != r.size():
            z = utils.to_var(next(self.z_generator)[0])
            gz = self.g(z)
            r = utils.to_var(next(self.r_generator)[0])
        return r, gz

    def define_d(self, config):
        self.phi = networks.get_d(config)
        self.d_optimizer = networks.get_optim(self.phi.parameters(),config.d_lr, config)

    def psi(self, y):
        return -self.phi(y)

    def calc_dloss(self, x, y, tx, ty, ux, vy, config):
        d_loss = -torch.mean(ux + vy)
        d_loss += losses.gp_loss(x, y, self.phi, config.lambda_gp, clamp=config.clamp)
        return d_loss


    def calc_gloss(self, x, y, ux, vy, config):
        return torch.mean(vy)

    def get_stats(self,  config):
        """print outs"""
        stats = OrderedDict()
        stats['loss/disc'] = self.d_loss
        stats['loss/gen'] = self.g_loss
        return stats

    def get_networks(self):
        nets = OrderedDict([('phi', self.phi)])
        nets['gen'] = self.g
        return nets
