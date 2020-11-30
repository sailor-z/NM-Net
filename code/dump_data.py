#!/usr/bin/env python3
# dump_data.py ---
#
# Filename: dump_data.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Mon Apr  2 18:33:34 2018 (-0700)
# Version:
# Package-Requires: ()
# URL:
# Doc URL:
# Keywords:
# Compatibility:
#
#

# Commentary:
#
#
#
#

# Change Log:
#
#
#
# Copyright (C)
# Visual Computing Group @ University of Victoria
# Computer Vision Lab @ EPFL

# Code:


from __future__ import print_function

import itertools
import multiprocessing as mp
import os
import pickle
import sys
import time

import numpy as np
import copy
import dataset
import cv2
from config import get_config
from data import loadFromDir
from geom import get_episqr, get_episym, get_sampsons, parse_geom
from six.moves import xrange
from utils import loadh5, saveh5

eps = 1e-10
des_thre = 0.5
geo_thre = 0.3
lambda_thre = 0.4
adjacency_num = 32
use3d = False
config = None

config, unparsed = get_config()


def dump_data_pair(args):
    dump_dir, idx, ii, jj, queue = args

    # queue for monitoring
    if queue is not None:
        queue.put(idx)

    dump_file = os.path.join(dump_dir, "idx_sort-{}-{}.h5".format(ii, jj))
 #   txt_dict = os.path.join(dump_dir, "idx_sort-{}-{}.txt".format(ii, jj))

    if not os.path.exists(dump_file):
        # Load descriptors for ii
        desc_ii = loadh5(
            os.path.join(dump_dir, "kp-aff-desc-{}.h5".format(ii)))["desc"]
        desc_jj = loadh5(
            os.path.join(dump_dir, "kp-aff-desc-{}.h5".format(jj)))["desc"]

        # compute decriptor distance matrix
        distmat = np.sqrt(
            np.sum(
                (np.expand_dims(desc_ii, 1) - np.expand_dims(desc_jj, 0))**2,
                axis=2))
        # Choose K best from N
        idx_sort = np.argsort(distmat, axis=1)[:, :config.obj_num_nn]
        idx_sort = (
            np.repeat(
                np.arange(distmat.shape[0])[..., None],
                idx_sort.shape[1], axis=1
            ),
            idx_sort
        )
        distmat = distmat[idx_sort]
        # Dump to disk
        dump_dict = {}
        dump_dict["idx_sort"] = idx_sort
        saveh5(dump_dict, dump_file)


def perspectiveTransform(kp, H):
    kp_h = np.ones((3, 1))
    kp_h[0] = kp[0]
    kp_h[1] = kp[1]
    prj_kp = np.matmul(H, kp_h)
    prj_kp = prj_kp / prj_kp[2]
    return prj_kp
   
def drawMatches(img1, img2, x, mask):

    # Create a new output image that concatenates the two images together
    # (a.k.a) a montage
    img1 = img1.transpose(1, 2, 0)
    img2 = img2.transpose(1, 2, 0)
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]

    out = np.zeros((max([rows1,rows2]), cols1+cols2 ,3), dtype='uint8')

    # Place the first image to the left
  #  out[:rows1,:cols1] = np.dstack([img1, img1, img1])
    out[:rows1,:cols1, :] = img1
    # Place the next image to the right of it
    out[:rows2,cols1:, :] = img2
    
    select = int(x.shape[1] / 10)
    # For each pair of points we have between both images
    # draw circles, then connect a line between them
    for i in range(select):
        # x - columns
        # y - rows
        (x1,y1) = x[0, i, :2]
        (x2,y2) = x[0, i, 2:]
        # Draw a small circle at both co-ordinates
        # radius 4
        # colour blue
        # thickness = 1
        '''
        a = np.random.randint(0,256)
        b = np.random.randint(0,256)
        c = np.random.randint(0,256)
        cv2.circle(out, (int(np.round(x1)),int(np.round(y1))), 2, (a, b, c), 1)     
        cv2.circle(out, (int(np.round(x2)+cols1),int(np.round(y2))), 2, (a, b, c), 1)
        cv2.line(out, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (a, b, c), 1, shift=0) 
        '''
        # Draw a line in between the two points
        # thickness = 1
        # colour blue
        
        if mask[i] == 0:
            cv2.circle(out, (int(np.round(x1)),int(np.round(y1))), 2, (0, 0, 255), 1)     
            cv2.circle(out, (int(np.round(x2)+cols1),int(np.round(y2))), 2, (0, 0, 255), 1)
            cv2.line(out, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (0, 0, 255), 5, shift=0) 
        else:
            cv2.circle(out, (int(np.round(x1)),int(np.round(y1))), 2, (0, 255, 0), 1)     
            cv2.circle(out, (int(np.round(x2)+cols1),int(np.round(y2))), 2, (0, 255, 0), 1)
            cv2.line(out, (int(np.round(x1)),int(np.round(y1))), (int(np.round(x2)+cols1),int(np.round(y2))), (0, 255, 0), 5, shift=0) 
        
    cv2.imwrite('/data04/zhaochen/debug/result.jpg', out)
    np.savetxt('/data04/zhaochen/debug/location.txt', x[0, :, :])
    np.savetxt('/data04/zhaochen/debug/mask.txt', mask)
    # Also return the image if you'd like a copy
    return out

def geometric_inlier_statistic(xs_initial, mask):
    # geometric k-nearest
    norm_input = (xs_initial / np.linalg.norm(xs_initial, ord=2, axis=-1, keepdims=True))
    index = np.argsort(-np.matmul(norm_input, norm_input.transpose(1, 0)), axis=-1)[:, 1:adjacency_num]
    inlier_ratio = []
    for i in range(xs_initial.shape[0]):
        inlier = 0
        for idx in range(adjacency_num - 1):
            if mask[index[i, idx]] == 1:
                inlier += 1
                
        inlier = inlier / (adjacency_num - 1)
        inlier_ratio.append(inlier)
    
    inlier_ratio = np.asarray(inlier_ratio)
    
    inlier_ratio = np.mean(inlier_ratio)
    gt_ratio = np.sum(mask) / xs_initial.shape[0]
    
    ratio = np.zeros((1, 2))
    ratio[0, 0] = inlier_ratio
    ratio[0, 1] = gt_ratio
    return ratio
    
def neighbors_inlier_statistic(xs_initial, mask, aff):    
    # neighbors k-nearest
    x = np.concatenate((xs_initial[:, :2], np.ones((xs_initial.shape[0], 1))), axis=-1)
    prj = []
    for i in range(x.shape[0]):
        affine_x = np.linalg.inv(aff[i][:9].reshape(3, 3))
        affine_y = aff[i][9:].reshape(3, 3)
        H = np.dot(affine_y, affine_x)
        y_prj = np.dot(H, x.transpose(1, 0))
        y_prj = (y_prj / np.expand_dims(y_prj[2, :], axis =0))[:2, :].transpose(1, 0)
        prj += [y_prj]
        
    prj_ = np.array(prj)
    prj_dis = np.zeros((x.shape[0], x.shape[0])).astype(np.float32)
    
    for i in range(x.shape[0]):
        prj_dis[i, :] = np.sum(np.abs(prj_[:, i, :] - np.expand_dims(prj_[i, i, :], axis=0)), axis=-1)
    prj_dis = prj_dis + prj_dis.transpose(1, 0)
    
    index = np.argsort(prj_dis, axis=-1)[:, 1:adjacency_num]
    
    inlier_ratio = []
    for i in range(xs_initial.shape[0]):
        inlier = 0
        for idx in range(adjacency_num - 1):
            if mask[index[i, idx]] == 1:
                inlier += 1
                
        inlier = inlier / (adjacency_num - 1)
        inlier_ratio.append(inlier)
    
    inlier_ratio = np.asarray(inlier_ratio)
    
    inlier_ratio = np.mean(inlier_ratio)
    gt_ratio = np.sum(mask) / xs_initial.shape[0]
    
    ratio = np.zeros((1, 2))
    ratio[0, 0] = inlier_ratio
    ratio[0, 1] = gt_ratio
    return ratio
    
def make_xy(pair_index, img, kp, desc, aff, K, R, t, cur_folder):
    kp_initial = copy.deepcopy(kp)
    xs = []
    xs_initial = []
    ys = []
    Rs = []
    ts = []
    img1s = []
    img2s = []
    cx1s = []
    cy1s = []
    f1s = []
    cx2s = []
    cy2s = []
    f2s = []
    affine = []
    # Create a random folder in scratch
    dump_dir = os.path.join(cur_folder, "dump")
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)
    
    pool_arg = []
    
    for idx in range(pair_index.shape[0]):
        ii = pair_index[idx, 0]
        jj = pair_index[idx, 1]

        print(
            "\rExtracting keypoints {} / {}".format(idx, pair_index.shape[0]),
            end="")
        sys.stdout.flush()
                
        # Check and extract keypoints if necessary
        for i in [ii, jj]:
            i = int(i)
            dump_file = os.path.join(dump_dir, "kp-aff-desc-{}.h5".format(i))
            if not os.path.exists(dump_file):
                # Correct coordinates using K
                cx = K[i, 2]
                cy = K[i, 5]

                # Correct focals
                fx = K[i, 0]
                fy = K[i, 4]
                
                kp[i] = (kp[i] - np.array([[cx, cy]])) / np.asarray([[fx, fy]])
     
                # Write descs to harddisk to parallize
                dump_dict = {}
                dump_dict["kp_normal"] = kp[i]
                dump_dict["kp"] = kp_initial[i]
                dump_dict["desc"] = desc[i]
                dump_dict["aff"] = aff[i]
                saveh5(dump_dict, dump_file)
            else:
                dump_dict = loadh5(dump_file)
                kp[i] = dump_dict["kp_normal"]
                kp_initial[i] = dump_dict["kp"]
                desc[i] = dump_dict["desc"]
                aff[i] = dump_dict["aff"]
                
        pool_arg += [(dump_dir, idx, int(ii), int(jj))]
    print("")
    
    # Run mp job
    ratio_CPU = 0.9
    number_of_process = int(ratio_CPU * mp.cpu_count())
    pool = mp.Pool(processes=number_of_process)
    manager = mp.Manager()
    queue = manager.Queue()
    for idx_arg in xrange(len(pool_arg)):
        pool_arg[idx_arg] = pool_arg[idx_arg] + (queue,)
    # map async
    pool_res = pool.map_async(dump_data_pair, pool_arg)
    # monitor loop
    while True:
        if pool_res.ready():
            break
        else:
            size = queue.qsize()
            print("\rDistMat {} / {}".format(size, len(pool_arg)), end="")
            sys.stdout.flush()
            time.sleep(1)
    pool.close()
    pool.join()
    print("")
    # Pack data
    idx = 0
    total_num = 0
    good_num = 0
    bad_num = 0
    ratio1 = []
    ratio2 = []
    
    for idx in range(pair_index.shape[0]):
        ii = int(pair_index[idx, 0])
        jj = int(pair_index[idx, 1])

        print("\rWorking on {} / {}".format(idx, pair_index.shape[0]), end="")
        sys.stdout.flush()
               
        K1 = K[ii].reshape(3, 3)
        K2 = K[jj].reshape(3, 3)
        
        # ------------------------------
        # Get dR
        R_i = R[ii].reshape(3, 3)
        R_j = R[jj].reshape(3, 3)
        dR = np.dot(R_j, R_i.T)
        # Get dt
        t_i = t[ii].reshape(3, 1)
        t_j = t[jj].reshape(3, 1)
        dt = t_j - np.dot(dR, t_i)
        # ------------------------------
        # Get keypoints for the first image
        x1 = kp[ii]
        y1 = np.concatenate((kp[ii], np.ones((kp[ii].shape[0], 1))), axis=1)
        # Project the first points into the second image
        y1p = np.matmul(dR[None], y1[..., None]) + dt[None]
        # move back to the canonical plane
        x1p = y1p[:, :2, 0] / y1p[:, 2, 0][..., None]
        # ------------------------------
        # Get keypoints for the second image
        x2 = kp[jj]
        # # DEBUG ------------------------------
        # # Check if the image projections make sense
        # draw_val_res(
        #     img[ii],
        #     img[jj],
        #     x1, x1p, np.random.rand(x1.shape[0]) < 0.1,
        #     (img[ii][0].shape[1] - 1.0) * 0.5,
        #     (img[ii][0].shape[0] - 1.0) * 0.5,
        #     parse_geom(geom, geom_type)["K"][ii, 0, 0],
        #     (img[jj][0].shape[1] - 1.0) * 0.5,
        #     (img[jj][0].shape[0] - 1.0) * 0.5,
        #     parse_geom(geom, geom_type)["K"][jj, 0, 0],
        #     "./debug_imgs/",
        #     "debug_img{:04d}.png".format(idx)
        # )
        # ------------------------------
        # create x1, y1, x2, y2 as a matrix combo
        x1mat = np.repeat(x1[:, 0][..., None], len(x2), axis=-1)
        y1mat = np.repeat(x1[:, 1][..., None], len(x2), axis=1)
        x1pmat = np.repeat(x1p[:, 0][..., None], len(x2), axis=-1)
        y1pmat = np.repeat(x1p[:, 1][..., None], len(x2), axis=1)
        x2mat = np.repeat(x2[:, 0][None], len(x1), axis=0)
        y2mat = np.repeat(x2[:, 1][None], len(x1), axis=0)
        # Load precomputed nearest neighbors
        idx_sort = loadh5(os.path.join(
            dump_dir, "idx_sort-{}-{}.h5".format(ii, jj)))["idx_sort"]
        # Move back to tuples
        idx_sort = (idx_sort[0], idx_sort[1])
        x1mat = x1mat[idx_sort]
        y1mat = y1mat[idx_sort]
        x1pmat = x1pmat[idx_sort]
        y1pmat = y1pmat[idx_sort]
        x2mat = x2mat[idx_sort]
        y2mat = y2mat[idx_sort]
        # Turn into x1, x1p, x2
        x1 = np.concatenate(
            [x1mat.reshape(-1, 1), y1mat.reshape(-1, 1)], axis=1)
        x1p = np.concatenate(
            [x1pmat.reshape(-1, 1),
             y1pmat.reshape(-1, 1)], axis=1)
        x2 = np.concatenate(
            [x2mat.reshape(-1, 1), y2mat.reshape(-1, 1)], axis=1)

        # make xs in NHWC
        xs += [
            np.concatenate([x1, x2], axis=1).T.reshape(4, 1, -1).transpose(
                (1, 2, 0))
        ]
        # ------------------------------
        # Get the geodesic distance using with x1, x2, dR, dt
        
        if config.obj_geod_type == "sampson":
            geod_d = get_sampsons(x1, x2, dR, dt)
        elif config.obj_geod_type == "episqr":
            geod_d = get_episqr(x1, x2, dR, dt)
        elif config.obj_geod_type == "episym":
            geod_d = get_episym(x1, x2, dR, dt)
         #   geod_d = get_episym(x1, x2, dR, dt, K1_inv, K2_inv, K1, K2)
        # Get *rough* reprojection errors. Note that the depth may be noisy. We
        # ended up not using this...
        reproj_d = np.sum((x2 - x1p)**2, axis=1)
        # count inliers and outliers
        total_num += len(geod_d)
        good_num += np.sum((geod_d < config.obj_geod_th))
        bad_num += np.sum((geod_d >= config.obj_geod_th))
        '''
        mask = np.zeros(len(geod_d))
        for i in range(len(geod_d)):
            if geod_d[i] < config.obj_geod_th:
                mask[i] = 1
        np.savetxt(os.path.join(dump_dir, "mask-{}-{}.txt".format(ii, jj)), mask)
        '''        
        ys += [np.stack([geod_d, reproj_d], axis=1)]
        
        # Save R, t for evaluation
        Rs += [np.array(dR).reshape(3, 3)]
        # normalize t before saving
        dtnorm = np.sqrt(np.sum(dt**2))
        assert (dtnorm > 1e-5)
        dt /= dtnorm
        ts += [np.array(dt).flatten()]

        # Save img1 and img2 for display
        img1s += [img[ii]]
        img2s += [img[jj]]
        
        cx = K[ii, 2]
        cy = K[ii, 5]
        cx1s += [cx]
        cy1s += [cy]
        cx = K[jj, 2]
        cy = K[jj, 5]
        cx2s += [cx]
        cy2s += [cy]

        fx = K[ii, 0]
        fy = K[ii, 4]
        if np.isclose(fx, fy):
            f = fx
        else:
            f = (fx, fy)
        f1s += [f]
        fx = K[jj, 0]
        fy = K[jj, 4]
        if np.isclose(fx, fy):
            f = fx
        else:
            f = (fx, fy)
        f2s += [f]

        # Generate xs_initial and T-transform
        x1 = kp_initial[ii]
        x2 = kp_initial[jj]

        aff1 = aff[ii]
        aff2 = aff[jj]

        aff1 = np.repeat(aff1[:, :][..., None], len(aff2), axis=-1).transpose(0, 2, 1)
        aff2 = np.repeat(aff2[:, :][None], len(aff1), axis=0)

        x1mat = np.repeat(x1[:, 0][..., None], len(x2), axis=-1)
        y1mat = np.repeat(x1[:, 1][..., None], len(x2), axis=1)
        x2mat = np.repeat(x2[:, 0][None], len(x1), axis=0)
        y2mat = np.repeat(x2[:, 1][None], len(x1), axis=0)
        idx_sort = loadh5(os.path.join(
            dump_dir, "idx_sort-{}-{}.h5".format(ii, jj)))["idx_sort"]
        # Move back to tuples
        idx_sort = (idx_sort[0], idx_sort[1])
        x1mat = x1mat[idx_sort]
        y1mat = y1mat[idx_sort]
        x2mat = x2mat[idx_sort]
        y2mat = y2mat[idx_sort]
        aff1 = aff1[idx_sort]
        aff2 = aff2[idx_sort]

        # Turn into x1, x1p, x2
        x1 = np.concatenate(
            [x1mat.reshape(-1, 1), y1mat.reshape(-1, 1)], axis=1)
        x2 = np.concatenate(
            [x2mat.reshape(-1, 1), y2mat.reshape(-1, 1)], axis=1)

        affine += [np.concatenate([aff1.reshape(-1, 9), aff2.reshape(-1, 9)], axis=1)]
        xs_initial += [np.concatenate([x1, x2], axis=1).T.reshape(4, 1, -1).transpose((1, 2, 0))]
          
    print("")
    # Do *not* convert to numpy arrays, as the number of keypoints may differ
    # now. Simply return it
    print(".... done")
    if total_num > 0:
        print(" Good pairs = {}, Total pairs = {}, Ratio = {}".format(
            good_num, total_num, float(good_num) / float(total_num)))
        print(" Bad pairs = {}, Total pairs = {}, Ratio = {}".format(
            bad_num, total_num, float(bad_num) / float(total_num)))

    res_dict = {}
    res_dict["xs"] = xs
    res_dict["xs_initial"] = xs_initial
    res_dict["affine"] = affine
    res_dict["ys"] = ys
    res_dict["Rs"] = Rs
    res_dict["ts"] = ts
    res_dict["img1s"] = img1s
    res_dict["cx1s"] = cx1s
    res_dict["cy1s"] = cy1s
    res_dict["f1s"] = f1s
    res_dict["img2s"] = img2s
    res_dict["cx2s"] = cx2s
    res_dict["cy2s"] = cy2s
    res_dict["f2s"] = f2s

    return res_dict

print("-------------------------DUMP-------------------------")

# Read conditions

crop_center = config.data_crop_center
data_folder = config.data_dump_prefix

# Now start data prep
print("Preparing data for {}".format(config.data_tr.split(".")[0]))

        
#  train_path = getattr(config, "data_dir_" + _set[:2]) + split + "/" 
train_path = getattr(config, "data_dir_tr")

# Create data dump directory name
data_names = getattr(config, "data_tr")
data_name = data_names.split(".")[0]
cur_folder = "/".join([
    data_folder,
    data_name,
    "numkp-{}".format(config.obj_num_kp)
])

if not os.path.exists(cur_folder):
    os.makedirs(cur_folder)

img, kp, desc, aff, K, R, t = loadFromDir(
    train_path, cur_folder,
    "-16x16",
    bUseColorImage=True,
    crop_center=crop_center,
    load_hessian = True)
    
if len(kp) == 0:
    kp = [None] * len(img)
if len(desc) == 0:
    desc = [None] * len(img)

pair_index = np.loadtxt(train_path + "pair_index.txt")

# Check if we've done this folder already.
print(" -- Waiting for the data_folder to be ready")
ready_file = os.path.join(cur_folder, "ready")
if not os.path.exists(ready_file):
    print(" -- No ready file {}".format(ready_file))
    print(" -- Generating data")
    
    # Make xy for this pair
    data_dict = make_xy(pair_index, img, kp, desc, aff, K, R, t, cur_folder)

    # Let's pickle and save data. Note that I'm saving them
    # individually. This was to have flexibility, but not so much
    # necessary.
    for var_name in data_dict:
    #    cur_var_name = var_name + "_" + _set[:2]
        cur_var_name = var_name + "_tr"
        out_file_name = os.path.join(cur_folder, cur_var_name) + ".pkl"
        with open(out_file_name, "wb") as ofp:
            pickle.dump(data_dict[var_name], ofp)

    # Mark ready
    with open(ready_file, "w") as ofp:
        ofp.write("This folder is ready\n")
else:
    print("Done!")
exit()

#
# dump_data.py ends here
