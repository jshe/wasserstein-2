import os
import time
import numpy as np
import utils
import pickle
import glob

from torch.backends import cudnn
from data_loader import get_loader, get_data
from w2_model import W2
from bot_model import BaryOT
from options import Options
from tensorboardX import SummaryWriter
from sklearn.neighbors import KNeighborsClassifier

def main():
    ## parse flags
    config = Options().parse()
    utils.print_opts(config)

    ## set up folders
    exp_dir = os.path.join(config.exp_dir, config.exp_name)
    model_dir = os.path.join(exp_dir, 'models')
    img_dir = os.path.join(exp_dir, 'images')
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    if config.solver == 'none':
        model = None
    else:
        if config.use_tbx:
            # remove old tensorboardX logs
            logs = glob.glob(os.path.join(exp_dir, 'events.out.tfevents.*'))
            if len(logs) > 0:
                os.remove(logs[0])
            tbx_writer = SummaryWriter(exp_dir)
        else:
            tbx_writer = None

        ## initialize data loaders/generators & model
        r_loader, z_loader = get_loader(config)
        if config.solver == 'w2':
            model = W2(config, r_loader, z_loader)
        elif config.solver == 'bary_ot':
            model = BaryOT(config, r_loader, z_loader)
        cudnn.benchmark = True
        networks = model.get_networks()
        utils.print_networks(networks)

        ## training
        ## stage 1 (dual stage) of bary_ot
        start_time = time.time()
        if config.solver == 'bary_ot':
            print("Starting: dual stage for %d iters." % config.dual_iters)
            for step in range(config.dual_iters):
                model.train_diter_only(config)
                if ((step+1) % 100) == 0:
                    stats = model.get_stats(config)
                    end_time = time.time()
                    stats['disp_time'] = (end_time - start_time) / 60.
                    start_time = end_time
                    utils.print_out(stats, step+1, config.dual_iters, tbx_writer)
            print("dual stage iterations complete.")

        ## main training loop of w1 / w2 or stage 2 (map stage) of bary-ot
        map_iters = config.map_iters if config.solver == 'bary_ot' else config.train_iters
        if config.solver == 'bary_ot':
            print("Starting: map stage for %d iters." % map_iters)
        else:
            print("Starting training...")
        for step in range(map_iters):
            model.train_iter(config)
            if ((step+1) % 100) == 0:
                stats = model.get_stats(config)
                end_time = time.time()
                stats['disp_time'] = (end_time - start_time) / 60.
                start_time = end_time
                utils.print_out(stats, step+1, map_iters, tbx_writer)
            if ((step+1) % 500) == 0:
                images = model.get_visuals(config)
                utils.visualize_iter(images, img_dir, step+1, config)
        print("Training complete.")
        networks = model.get_networks()
        utils.save_networks(networks, model_dir)

    ## testing
    ## 1) classification accuracy
    print("Calculating domain adaptation accuracy...")
    utils.print_accuracy(config, model)

    ## 2) visualization
    if config.solver != 'none':
        root = "./usps_test" if config.direction == 'usps-mnist' else "./mnist_test"
        file = open(os.path.join(root, "data.pkl"), "rb")
        fixed_z = pickle.load(file)
        file.close()
        fixed_z = utils.to_var(fixed_z)
        fixed_gz = model.g(fixed_z).view(*fixed_z.size())
        utils.visualize_single(fixed_gz, os.path.join(img_dir, 'test.png'), config)

if __name__ == '__main__':
    main()
