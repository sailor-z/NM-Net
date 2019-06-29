#!/usr/bin/env python3
from config import get_config, print_usage
from pytorch_test import test_process
from tqdm import trange
import numpy as np
import torch
from model import NM_Net
import loss
import math
from torch import optim
import os
import sys
import torch.utils.data as Data
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.utils as utils
import matplotlib.pyplot as plt
from dataset import Data_Loader

config = None


def weights_init(m):
    if isinstance(m, torch.nn.Conv2d):
        torch.nn.init.xavier_normal_(m.weight.data)
        torch.nn.init.constant_(m.bias.data, 0)


def skew_symmetric(v):
    zero = np.zeros((len(v), 1))

    M = np.hstack((zero, -v[:, 2, :], v[:, 1, :],
                   v[:, 2, :], zero, -v[:, 0, :],
                   -v[:, 1, :], v[:, 0, :], zero))
    return M

def adjust_learning_rate(optimizer, epoch, opt):
    """Sets the learning rate to the initial LR decayed by 10 every 500 epochs"""
    lr = max((opt.lr * (opt.decay_rate ** (epoch  // opt.decay_step))), 1e-5)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    
def main(config):
    """The main function."""
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"
    database = sys.argv[-1]

    database_list = []
    if database == 'COLMAP':
        database_list += ["south"]
        database_list += ["gerrard"]
        database_list += ["graham"]
        database_list += ["person"]
    elif database == 'NARROW':
        database_list += ["lib-narrow"]
        database_list += ["mao-narrow"]
        database_list += ["main-narrow"]
        database_list += ["science-narrow"]
    elif database == 'WIDE':
        database_list += ["lib-wide"]
        database_list += ["mao-wide"]
        database_list += ["main-wide"]
        database_list += ["science-wide"]
    else:
        print("Input error!!")
        exit()

    log_dir = "log/"
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Initialize network
    print("Initializing...")
    Network = NM_Net().cuda()
    Network.apply(weights_init)
    Network.train()    
    
    d = Data_Loader(config, database, database_list, "train", initialize = False)
    
    data = Data.DataLoader(d, batch_size=config.train_batch_size, shuffle=True, num_workers=16, drop_last=True)
    loss_func = loss.Loss_classi().cuda()
    
    loss_his = []
    var_list = []

    optimizer = optim.Adam(Network.parameters(), lr=config.train_lr)
    scheduler = CosineAnnealingLR(optimizer, config.epochs, eta_min=config.train_lr*0.01)
    
    step = 0
    best_va_res = 0

    # ----------------------------------------
    # The training loop       
    for epoch in range(config.epochs):   
        loss_list = []            
        for i, (xs, Es, index, label) in enumerate(data, 0):
            xs = xs.cuda()
            Es = Es.cuda()
            index = index.cuda()
            label = label.cuda()
            output, weight = Network(xs, index)
            
            optimizer.zero_grad()
            
            l = loss_func(output, label)
            
            l.backward()
            optimizer.step()

            loss_list += [l]

        loss_list = torch.stack(loss_list).view(-1)
        print('Epoch: {} / {} ---- Trainning Loss : {}'.format(epoch, config.epochs, loss_list.mean()))

        # Write summary and save current model
        # ----------------------------------------
        torch.save(Network, log_dir + '/NM-Net_state.pth')
        # Validation
        va_res = test_process("valid", log_dir, '/NM-Net_state.pth', database, config, config.knn_num)
        
        print('Validation F-measure : {}'.format(va_res))

        var_list.append(va_res)

        np.savetxt(log_dir + './validation_list.txt', np.array(var_list))

        # Higher the better
        if va_res > best_va_res:
            print("Saving best model with va_res = {}".format(va_res))
            best_va_res = va_res
            # Save best validation result
            np.savetxt(log_dir + "/best_results.txt", best_va_res)
            # Save best model
            torch.save(Network, log_dir + '/NM-Net_best_state.pth')

    te_res = test_process("test", log_dir, '/NM-Net_best_state.pth', database, config, config.knn_num)

    print('Testing F-measure : {}'.format(te_res))

    np.savetxt(log_dir + "/test_results.txt", te_res)

if __name__ == "__main__":

    # ----------------------------------------
    config, unparsed = get_config()

    main(config)

#
# main.py ends here
