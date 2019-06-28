#!/usr/bin/env python3
import datetime
import os
import time
import torch
from model import NM_Net
import math
from dataset import Data_Loader
import pickle
import numpy as np
from torch.autograd import Variable
import tensorflow as tf
import torch.utils.data as Data
import cv2
from six.moves import xrange
from transformations import quaternion_from_matrix

min_kp_num = 500
margin = 0.05
def evaluate_R_t(R_gt, t_gt, R, t, q_gt=None):

    # from Utils.transformations import quaternion_from_matrix

    t = t.flatten()
    t_gt = t_gt.flatten()

    eps = 1e-15

    if q_gt is None:
        q_gt = quaternion_from_matrix(R_gt)
    q = quaternion_from_matrix(R)
    q = q / (np.linalg.norm(q) + eps)
    q_gt = q_gt / (np.linalg.norm(q_gt) + eps)
    loss_q = np.maximum(eps, (1.0 - np.sum(q * q_gt)**2))
    err_q = np.arccos(1 - 2 * loss_q)

    # dR = np.dot(R, R_gt.T)
    # dt = t - np.dot(dR, t_gt)
    # dR = np.dot(R, R_gt.T)
    # dt = t - t_gt
    t = t / (np.linalg.norm(t) + eps)
    t_gt = t_gt / (np.linalg.norm(t_gt) + eps)
    loss_t = np.maximum(eps, (1.0 - np.sum(t * t_gt)**2))
    err_t = np.arccos(np.sqrt(1 - loss_t))

    if np.sum(np.isnan(err_q)) or np.sum(np.isnan(err_t)):
        # This should never happen! Debug here
        import IPython
        IPython.embed()

    return err_q, err_t

def eval_nondecompose(p1s, p2s, E_hat, dR, dt, scores):

    # Use only the top 10% in terms of score to decompose, we can probably
    # implement a better way of doing this, but this should be just fine.
    num_top = len(scores) // 10
    num_top = max(1, num_top)
    th = np.sort(scores)[::-1][num_top]
    mask = scores >= th

    p1s_good = p1s[mask]
    p2s_good = p2s[mask]

    # Match types
    E_hat = E_hat.reshape(3, 3).astype(p1s.dtype)

    if p1s_good.shape[0] >= 5:
        # Get the best E just in case we get multipl E from findEssentialMat
        num_inlier, R, t, mask_new = cv2.recoverPose(
            E_hat, p1s_good, p2s_good)
        try:
            err_q, err_t = evaluate_R_t(dR, dt, R, t)
        except:
            print("Failed in evaluation")
            print(R)
            print(t)
            err_q = np.pi
            err_t = np.pi / 2
    else:
        err_q = np.pi
        err_t = np.pi / 2

    loss_q = np.sqrt(0.5 * (1 - np.cos(err_q)))
    loss_t = np.sqrt(1.0 - np.cos(err_t)**2)

    # Change mask type
    mask = mask.flatten().astype(bool)

    mask_updated = mask.copy()
    if mask_new is not None:
        # Change mask type
        mask_new = mask_new.flatten().astype(bool)
        mask_updated[mask] = mask_new
    if err_q == 0:
        print(err_t)
    return err_q, err_t, loss_q, loss_t, np.sum(num_inlier), mask_updated

def estimate_E(input, weight):
    E = []
    data = []
    data.append(torch.unsqueeze(input[:, 0, :, 0], -1))  # u1
    data.append(torch.unsqueeze(input[:, 0, :, 1], -1)) # v1
    data.append(torch.unsqueeze(input[:, 0, :, 2], -1))  # u2
    data.append(torch.unsqueeze(input[:, 0, :, 3], -1))
    data.append(torch.unsqueeze(torch.ones(input.size(0),input.size(2)).cuda(), -1))
    X = torch.cat((data[2] * data[0], data[2] * data[1], data[2], data[3] * data[0], data[3] * data[1],
                       data[3], data[0], data[1], data[4]), -1)

    W = torch.stack([torch.diag(weight[i]) for i in range(1)])
    M = torch.bmm(X.transpose(1, 2), W)
    M = torch.bmm(M, X)
    svd = torch.stack([torch.svd(M[i])[2][:, 8] for i in range(1)])
    E = (svd / (torch.norm(svd, 2, dim=-1, keepdim=True) + 1e-6))
    return E

def CV_estimate_E(input, weight):
    try:
        inlier = input.squeeze()[weight.squeeze() > 0].cpu().data.numpy()
        x = inlier[:, :2]
        y = inlier[:, 2:]
        e, mask = cv2.findFundamentalMat(x, y, cv2.FM_8POINT)
        e = np.reshape(e, (1, 9))
        e = (e / np.linalg.norm(e, ord=2, axis=1, keepdims=True))
        return e, 1
    except (IndexError, ValueError):
        return 0, 0
        
def local_consistency(xs_initial, affine):  #16*1*2000*4 16*2000*18
    x = xs_initial[:, 0, :, 0:2]
    y = xs_initial[:, 0, :, 2:4]
    affine = affine.view(-1, 18)    
    affine_x = torch.stack([torch.inverse(affine[i][:9].view(3, 3)) for i in range (affine.shape[0])])
    affine_y = affine[:, 9:].view(-1, 3, 3)
        
    H = torch.bmm(affine_y, affine_x).view(xs_initial.size(0), xs_initial.size(2), 3, 3)
    
    ones = torch.ones(x.size(0), x.size(1), 1)
    x = torch.cat((x, ones), dim=-1)
    y = torch.cat((y, ones), dim=-1)
    index = []
    for i in range(x.size(0)):
        x_repeat = x[i].t().repeat(x[i].size(0), 1, 1)
        y_prj = torch.bmm(H[i], x_repeat)
        y_prj = (y_prj / y_prj[:, 2, :].unsqueeze(1))[:, 0:2, :]
        
        prj_dis = torch.stack([torch.sum(torch.pow((y_prj[idx] - y_prj[idx, :, idx].unsqueeze(-1)), 2), dim=0) for idx in range(x[i].size(0))])
        prj_dis = prj_dis + prj_dis.t()
        index.append(torch.sort(prj_dis, dim=1, descending=False)[1][:, :8]) 
    return torch.stack(index)
    
def local_score(xs_initial, affine):  #16*1*2000*4 16*2000*18
    x = xs_initial[:, 0, :, 0:2]
    y = xs_initial[:, 0, :, 2:4]
    affine = affine.view(-1, 18)    
    affine_x = torch.stack([torch.inverse(affine[i][:9].view(3, 3)) for i in range (affine.shape[0])])
    affine_y = affine[:, 9:].view(-1, 3, 3)
        
    H = torch.bmm(affine_y, affine_x).view(xs_initial.size(0), xs_initial.size(2), 3, 3)
    
    ones = torch.ones(x.size(0), x.size(1), 1)
    x = torch.cat((x, ones), dim=-1)
    y = torch.cat((y, ones), dim=-1)
    score = []
    for i in range(x.size(0)):
        x_repeat = x[i].t().repeat(x[i].size(0), 1, 1)
        y_prj = torch.bmm(H[i], x_repeat)
        y_prj = (y_prj / y_prj[:, 2, :].unsqueeze(1))[:, 0:2, :]
        
        prj_dis = torch.stack([torch.sum(torch.pow((y_prj[idx] - y_prj[idx, :, idx].unsqueeze(-1)), 2), dim=0) for idx in range(x[i].size(0))])
        prj_dis = prj_dis + prj_dis.t()
        score.append(torch.sort(prj_dis, dim=1, descending=False)[0][:, :8]) 
    score = torch.stack(score)
    score = (-margin * score).exp()
    
    return score  
  
def test_process(mode, save_file_cur, model_name, data_name, config, adjacency_num=8):
    
    d = Data_Loader(config, data_name, None, mode, initialize = False)
    
    data = Data.DataLoader(d, batch_size=1, shuffle=False, num_workers=16, drop_last=False)

    Network = torch.load(save_file_cur + model_name).cuda()
    Network.eval()

    P = []
    R = []
    F = []
    MSE = []
    MAE = []
                       
    for i, (xs, Es, index, label) in enumerate(data, 0):
        xs = xs.cuda()
        Es = Es.cuda()
        index = index.cuda()
        label = label.cuda()
        output, weight = Network(xs, index)

        label = label.type(torch.FloatTensor)
        mask = (weight > 0).type(torch.FloatTensor)

        p = torch.sum(mask * label) / torch.sum(mask)
        if math.isnan(p):
            p = torch.Tensor([0])
        r = torch.sum(mask * label) / torch.sum(label) 
        if math.isnan(r):
            r = torch.Tensor([0])
        f = 2 * p * r / (p + r)
        if math.isnan(f):
            f = torch.Tensor([0]) 
        P.append(p.cpu().numpy())
        R.append(r.cpu().numpy())
        F.append(f.cpu().numpy())
        
        if mode == 'test':
            E_gt = np.array(Es.cpu().numpy()).reshape(1, 9).astype(np.float32)
            E, m = CV_estimate_E(xs, weight)
            if m == 0:
                continue
            E = E.reshape(1, 9)
            mse = np.sum(np.power(E_gt - E, 2), axis = -1)
            mae = np.sum(np.abs(E_gt - E), axis = -1)
            MSE.append(mse.mean())
            MAE.append(mae.mean())

    p_ = np.expand_dims(np.mean(np.array(P)), axis=0)
    r_ = np.expand_dims(np.mean(np.array(R)), axis=0)
    f_ = np.expand_dims(np.mean(np.array(F)), axis=0)

    log_path = os.path.join(save_file_cur, mode)
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    np.savetxt(os.path.join(log_path, "Precision.txt"), p_ * 100)
    np.savetxt(os.path.join(log_path, "Recall.txt"), r_ * 100)
    np.savetxt(os.path.join(log_path, "F-measure.txt"), f_ * 100)
    if mode == 'test':
        mse = np.expand_dims(np.mean(np.array(MSE)), axis=0)
        mae = np.expand_dims(np.mean(np.array(MAE)), axis=0)
        median = np.expand_dims(np.median(np.array(MAE)), axis=0)
        Max = np.expand_dims(np.max(np.array(MAE)), axis=0)
        Min = np.expand_dims(np.min(np.array(MAE)), axis=0)
        np.savetxt(os.path.join(log_path, "MSE.txt"), mse)
        np.savetxt(os.path.join(log_path, "MAE.txt"), mae)
        np.savetxt(os.path.join(log_path, "Median.txt"), median)
        np.savetxt(os.path.join(log_path, "Max.txt"), Max)
        np.savetxt(os.path.join(log_path, "Min.txt"), Min)

    return f_
 