from __future__ import print_function
import pickle
import numpy as np
import torch
import os
import sys
import multiprocessing as mp
import time
import random
from tqdm import trange
from torch.utils.data import Dataset
max_kp_num = 2000
margin = 1e-3

def skew_symmetric(v):
    zero = np.zeros((len(v), 1))

    M = np.hstack((zero, -v[:, 2, :], v[:, 1, :],
        v[:, 2, :], zero, -v[:, 0, :],
        -v[:, 1, :], v[:, 0, :], zero))
    return M

def parallel(config, x, affine, y):  
    c_num = x.shape[1]
    
    prj = []
    ys = np.zeros(y.shape, dtype=np.float32)
    for i in range(c_num):
        try:
            affine_x = np.linalg.inv(affine[i][:9].reshape(3, 3))
            affine_y = affine[i][9:].reshape(3, 3)
            H = np.dot(affine_y, affine_x)
            y_prj = np.dot(H, x)
            y_prj = (y_prj / np.expand_dims(y_prj[2, :], axis =0))[:2, :].transpose(1, 0)
            prj += [y_prj]
            ys[i] = y[i]

        except np.linalg.linalg.LinAlgError:
            ys[i] = [1, 1]
            continue
        
    prj_ = np.array(prj)
    
    prj_dis = np.zeros([len(prj), len(prj)], dtype=np.float32)

    for i in range(len(prj)):
        prj_dis[i, :] = np.sum(np.abs(prj_[:, i, :] - np.expand_dims(prj_[i, i, :], axis=0)), axis=-1)
    prj_dis = prj_dis + prj_dis.transpose(1, 0)

    index = np.zeros((len(prj), config.knn_num)).astype(np.int64)
    score = np.zeros((len(prj), config.knn_num)).astype(np.float32)

    for j in range(len(prj)):
        index[j, :] = np.argsort(prj_dis[j])[:config.knn_num]
        score[j, :] = np.exp(-np.sort(prj_dis[j])[:config.knn_num] * margin)
    
    if len(prj) < c_num:
        zeros = np.zeros([c_num - len(prj), config.knn_num], dtype=np.float32)
        score = np.concatenate([score, zeros], axis=0)
        pad_index = np.arange(len(prj), c_num)
        pad_index = np.expand_dims(pad_index, axis=-1).repeat(config.knn_num, axis=-1)
        index = np.concatenate([index, pad_index], axis=0)
    return index, score, ys
        
def local_score(config, xs_initial, affine, ys): 
    x = xs_initial[0, :, 0:2]
    c_num = x.shape[0]
    ones = np.ones((c_num, 1))
    x = np.concatenate((x, ones), axis=-1).transpose(1, 0).astype(np.float32)

    index, score, y = parallel(config, x, affine, ys)
    return index, score, y
    
def data_initialization(config, database, data_list, score_idx=False):
    
    
    # Now load data.
    var_name_list = [
        "xs", "xs_initial", "ys", "Rs", "ts", "affine"
    ]
    data_folder = config.data_dump_prefix

    # Let's unpickle and save data
    data_name = []
    for data in data_list:
        data_name += [os.path.join(data_folder, data, "numkp-{}".format(config.obj_num_kp))]
    
    data = {}
    for cur_folder in data_name:

        ready_file = os.path.join(cur_folder, "ready")
        if not os.path.exists(ready_file):
            raise RuntimeError("Data is not prepared!")
                    
        for var_name in var_name_list:
            cur_var_name = var_name + "_tr"
            in_file_name = os.path.join(cur_folder, cur_var_name) + ".pkl"

            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] = pickle.load(ifp)

  #  e_gt_unnorm = skew_symmetric(np.expand_dims(np.array(data["ts"]), axis=-1))
    e_gt_unnorm = np.reshape(np.matmul(
                np.reshape(skew_symmetric(np.expand_dims(np.array(data["ts"]), axis=-1)), (len(data["ts"]), 3, 3)),
                np.reshape(np.array(data["Rs"]), (len(data["ts"]), 3, 3))), (len(data["ts"]), 9))
    e_gt = e_gt_unnorm / np.linalg.norm(e_gt_unnorm, ord=2, axis=1, keepdims=True)
    data["Es"] = e_gt

    xs = []
    xs_initial = []
    ys = []
    Rs = []
    ts = []
    affine = []
    Es = []   
    index = []
    score = []

    for i in trange(len(data["xs"])):
        xs += [data["xs"][i]]
        xs_initial += [data["xs_initial"][i]]
        Rs += [data["Rs"][i]]
        ts += [data["ts"][i]]
        affine += [data["affine"][i]]
        Es += [data["Es"][i]]

        if score_idx==True: 
            idx, sco, y = local_score(config, data["xs_initial"][i], data["affine"][i], data["ys"][i])
            ys += [y]
            index += [idx]
            score += [sco]    
 
    shuffle_list = list(zip(xs, xs_initial, ys, Rs, ts, affine, Es, index, score)) 
    random.shuffle(shuffle_list)
    
    xs, xs_initial, ys, Rs, ts, affine, Es, index, score = zip(*shuffle_list)

    var_name_list = ["xs", "xs_initial", "ys", "Rs", "ts", "affine", "Es", "index", "score"]
    
    data = {}    
    data["xs"] = xs[:int(0.7 * len(xs))]
    data["xs_initial"] = xs_initial[:int(0.7 * len(xs))]
    data["ys"] = ys[:int(0.7 * len(xs))]
    data["Rs"] = Rs[:int(0.7 * len(xs))]
    data["ts"] = ts[:int(0.7 * len(xs))]
    data["affine"] = affine[:int(0.7 * len(xs))]
    data["Es"] = Es[:int(0.7 * len(xs))]
    data["index"] = index[:int(0.7 * len(xs))]
    data["score"] = score[:int(0.7 * len(xs))]

    print('Size of training data', len(data["xs"]))
    
    train_data_path = os.path.join(data_folder, database, "train")
    if not os.path.exists(train_data_path):
        os.makedirs(train_data_path)

    for var_name in var_name_list:
        in_file_name = os.path.join(train_data_path, var_name) + ".pkl"
        with open(in_file_name, "wb") as ofp:
            pickle.dump(data[var_name], ofp)
            
    data = {}    
    data["xs"] = xs[int(0.7 * len(xs)) : int(0.85 * len(xs))]
    data["xs_initial"] = xs_initial[int(0.7 * len(xs)) : int(0.85 * len(xs))]
    data["ys"] = ys[int(0.7 * len(xs)) : int(0.85 * len(xs))]
    data["Rs"] = Rs[int(0.7 * len(xs)) : int(0.85 * len(xs))]
    data["ts"] = ts[int(0.7 * len(xs)) : int(0.85 * len(xs))]
    data["affine"] = affine[int(0.7 * len(xs)) : int(0.85 * len(xs))]
    data["Es"] = Es[int(0.7 * len(xs)) : int(0.85 * len(xs))]
    data["index"] = index[int(0.7 * len(xs)) : int(0.85 * len(xs))]
    data["score"] = score[int(0.7 * len(xs)) : int(0.85 * len(xs))]
    
    print('Size of validation data', len(data["xs"]))
    
    valid_data_path = os.path.join(data_folder, database, "valid")
    if not os.path.exists(valid_data_path):
        os.makedirs(valid_data_path)

    for var_name in var_name_list:
        in_file_name = os.path.join(valid_data_path, var_name) + ".pkl"
        with open(in_file_name, "wb") as ofp:
            pickle.dump(data[var_name], ofp)
            
    data = {}    
    data["xs"] = xs[int(0.85 * len(xs)) : len(xs)]
    data["xs_initial"] = xs_initial[int(0.85 * len(xs)) : len(xs)]
    data["ys"] = ys[int(0.85 * len(xs)) : len(xs)]
    data["Rs"] = Rs[int(0.85 * len(xs)) : len(xs)]
    data["ts"] = ts[int(0.85 * len(xs)) : len(xs)]
    data["affine"] = affine[int(0.85 * len(xs)) : len(xs)]
    data["Es"] = Es[int(0.85 * len(xs)) : len(xs)]
    data["index"] = index[int(0.85 * len(xs)) : len(xs)]
    data["score"] = score[int(0.85 * len(xs)) : len(xs)]

    print('Size of testing data', len(data["xs"]))
    
    test_data_path = os.path.join(data_folder, database, "test")
    if not os.path.exists(test_data_path):
        os.makedirs(test_data_path)

    for var_name in var_name_list:
        in_file_name = os.path.join(test_data_path, var_name) + ".pkl"
        with open(in_file_name, "wb") as ofp:
            pickle.dump(data[var_name], ofp)
    
   
def load_data(config, data_name, var_mode):    
    print("Loading {} data".format(var_mode))
    data_folder = config.data_dump_prefix
    data_path = os.path.join(data_folder, data_name, var_mode)
    var_name_list = ["xs", "xs_initial", "ys", "Rs", "ts", "Es", "affine", "index", "score"]               
    data = {}
    
    for var_name in var_name_list:
        in_file_name = os.path.join(data_path, var_name) + ".pkl"
        with open(in_file_name, "rb") as ifp:
            if var_name in data:
                data[var_name] += pickle.load(ifp)
            else:
                data[var_name] = pickle.load(ifp)
    return data

class Data_Loader(Dataset):
    def __init__(self, config, database, data_list, var_mode, initialize):
        super(Data_Loader, self).__init__()
        
        self.config = config
        self.adjacency_num = config.knn_num
        self.var_mode = var_mode
        self.database = database
        self.data_list = data_list
        self.initialize = initialize

        if self.initialize == True:
            data_initialization(config, self.database, self.data_list, score_idx=True)

        self.data = load_data(self.config, self.database, self.var_mode)
            
    def __getitem__(self, item):
        xs = torch.from_numpy(self.data["xs"][item]).type(torch.FloatTensor)
        ys = torch.from_numpy(self.data["ys"][item]).type(torch.FloatTensor)
        Es = torch.from_numpy(self.data["Es"][item]).type(torch.FloatTensor)   
        index = torch.from_numpy(self.data["index"][item][:, :self.adjacency_num])
        label = (ys[:, 0] < self.config.obj_geod_th).type(torch.FloatTensor)  
        return (xs, Es, index, label)

    def __len__(self):
        return len(self.data["xs"])
