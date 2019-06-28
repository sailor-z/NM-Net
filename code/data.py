# fundatmental_toy.py ---
#
# Filename: fundatmental_toy.py
# Description:
# Author: Kwang Moo Yi
# Maintainer:
# Created: Tue Feb 14 10:59:18 2017 (+0100)
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

# Load data

from __future__ import print_function

import os
import pickle

import h5py
import numpy as np
import matlab.engine
import cv2
from transformations import quaternion_from_matrix
from utils import loadh5, saveh5

min_kp_num = 100

def load_geom(geom_file, geom_type, scale_factor, flip_R=False):
    if geom_type == "calibration":
        # load geometry file
        geom_dict = loadh5(geom_file)
        # Check if principal point is at the center
        K = geom_dict["K"]
        # assert(abs(K[0, 2]) < 1e-3 and abs(K[1, 2]) < 1e-3)
        # Rescale calbration according to previous resizing
        S = np.asarray([[scale_factor, 0, 0],
                        [0, scale_factor, 0],
                        [0, 0, 1]])
        K = np.dot(S, K)
        geom_dict["K"] = K
        # Transpose Rotation Matrix if needed
        if flip_R:
            R = geom_dict["R"].T.copy()
            geom_dict["R"] = R
        # append things to list
        geom_list = []
        geom_info_name_list = ["K", "R", "T", "imsize"]
        for geom_info_name in geom_info_name_list:
            geom_list += [geom_dict[geom_info_name].flatten()]
        # Finally do K_inv since inverting K is tricky with theano
        geom_list += [np.linalg.inv(geom_dict["K"]).flatten()]
        # Get the quaternion from Rotation matrices as well
        q = quaternion_from_matrix(geom_dict["R"])
        geom_list += [q.flatten()]
        # Also add the inverse of the quaternion
        q_inv = q.copy()
        np.negative(q_inv[1:], q_inv[1:])
        geom_list += [q_inv.flatten()]
        # Add to list
        geom = np.concatenate(geom_list)

    elif geom_type == "homography":
        H = np.loadtxt(geom_file)
        geom = H.flatten()

    return geom

def loadFromDir(train_data_dir, cur_folder, gt_div_str="", bUseColorImage=True,
                input_width=512, crop_center=False, load_hessian=True):
    """Loads data from directory.

    train_data_dir : Directory containing data

    gt_div_str : suffix for depth (e.g. -8x8)

    bUseColorImage : whether to use color or gray (default false)

    input_width : input image rescaling size

    """

    # read the list of imgs and the homography
    train_data_dir = train_data_dir.rstrip("/") + "/"
    img_list_file = train_data_dir + "list.txt"
    K_list_file = train_data_dir + "models/K/K.txt"
    R_list_file = train_data_dir + "models/R/R.txt"
    t_list_file = train_data_dir + "models/t/t.txt"

    # parse the file
    image_fullpath_list = []
    with open(img_list_file, "r") as img_list:
        while True:
            # read a single line
            tmp = img_list.readline()
            if type(tmp) != str:
                line2parse = tmp.decode("utf-8")
            else:
                line2parse = tmp
            if not line2parse:
                break
            # strip the newline at the end and add to list with full path
            image_fullpath_list += [train_data_dir + line2parse.rstrip("\n")]
    
    K = np.loadtxt(K_list_file)
    R = np.loadtxt(R_list_file)
    t = np.loadtxt(t_list_file)         
    # For each image and geom file in the list, read the image onto
    # memory. We may later on want to simply save it to a hdf5 file
    x = []
    kp = []
    desc = []
    img_name = []
    aff = []
    idxImg = 0
    for img_file in zip(image_fullpath_list):
        idxImg +=1
        print('\r -- Loading Image {} / {}'.format(
            idxImg, len(image_fullpath_list)
        ), end="")
        img_file = img_file[0]
        img_file = img_file.split('\\')[0] + '/' + img_file.split('\\')[1]
        # ---------------------------------------------------------------------
        # Read the color image
        if not bUseColorImage:
            # If there is not gray image, load the color one and convert to
            # gray
            if os.path.exists(img_file.replace(
                    "image_color", "image_gray"
            )):
                img = cv2.imread(img_file.replace(
                    "image_color", "image_gray"
                ), 0)
                assert len(img.shape) == 2
            else:
                # read the image
                img = cv2.cvtColor(cv2.imread(img_file),
                                   cv2.COLOR_BGR2GRAY)
            if len(img.shape) == 2:
                img = img[..., None]
            in_dim = 1

        else:
            img = cv2.imread(img_file)
            in_dim = 3
        assert(img.shape[-1] == in_dim)

        # Crop center and resize image into something reasonable
        if crop_center:
            rows, cols = img.shape[:2]
            if rows > cols:
                cut = (rows - cols) // 2
                img_cropped = img[cut:cut + cols, :]
            else:
                cut = (cols - rows) // 2
                img_cropped = img[:, cut:cut + rows]
            scale_factor = float(input_width) / float(img_cropped.shape[0])
            img = cv2.resize(img_cropped, (input_width, input_width))
        else:
            scale_factor = 1.0

        if load_hessian == True:  
            if os.path.exists(os.path.join(cur_folder, "dump/kp-aff-desc-{}.h5".format(idxImg-1))):
                dump_dict = loadh5(os.path.join(cur_folder, "dump/kp-aff-desc-{}.h5".format(idxImg-1)))
                x += [img.transpose(2, 0, 1)]
                kp += [dump_dict["kp"]]
                aff += [dump_dict["aff"]]
                desc += [dump_dict["desc"]]
            
            else:
                eng = matlab.engine.start_matlab()

                frames = eng.hessian(img_file, nargout=2) 
                keypoints = np.array(frames[0]).transpose()
                patches = np.array(frames[1]).transpose()
                
                x += [img.transpose(2, 0, 1)]
                
                keypoint = keypoints[:, :2]
                kp += [keypoint]
                
                affine = keypoints[:, 2:6].reshape(-1, 2, 2)
                affine = np.concatenate([affine, np.expand_dims(keypoint, axis=-1)], axis=-1)
                ones = np.ones([keypoints.shape[0], 1, 3], dtype=np.float)
                affine = np.concatenate([affine, ones], axis=1)

                aff += [affine.reshape(-1 , 9)]
    
                desc += [patches]
                
    np.savetxt(img_file.split('images')[0] + "image_index.txt", img_name, fmt='%s')

    return (x, kp, desc, aff, K, R, t)
            
def load_data(config, var_mode):
    """Main data loading routine"""

    print("Loading {} data".format(var_mode))

    # use only the first two characters for shorter abbrv
    var_mode = var_mode[:2]

    # Now load data.
    var_name_list = [
        "xs", "ys", "Rs", "ts",
        "img1s", "cx1s", "cy1s", "f1s",
        "img2s", "cx2s", "cy2s", "f2s",
    ]

    data_folder = config.data_dump_prefix
    if config.use_lift:
        data_folder += "_lift"

    # Let's unpickle and save data
    data = {}
    data_names = getattr(config, "data_" + var_mode)
    data_names = data_names.split(".")
    for data_name in data_names:
        cur_data_folder = "/".join([
            data_folder,
            data_name,
            "numkp-{}".format(config.obj_num_kp),
            "nn-{}".format(config.obj_num_nn),
        ])
        if not config.data_crop_center:
            cur_data_folder = os.path.join(cur_data_folder, "nocrop")
        suffix = "{}-{}".format(
            var_mode,
            getattr(config, "train_max_" + var_mode + "_sample")
        )
        cur_folder = os.path.join(cur_data_folder, suffix)
        ready_file = os.path.join(cur_folder, "ready")
        if not os.path.exists(ready_file):
            # data_gen_lock.unlock()
            raise RuntimeError("Data is not prepared!")

        for var_name in var_name_list:
            cur_var_name = var_name + "_" + var_mode
            in_file_name = os.path.join(cur_folder, cur_var_name) + ".pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] = pickle.load(ifp)

    return data
