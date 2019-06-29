# NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences (CVPR 2019 oral)
***

This repository is a reference implementation for Chen Zhao, Zhiguo Cao, Chi Li, Xin Li, and Jiaqi Yang, "NM-Net: Mining Reliable Neighbors for Robust Feature Correspondences", CVPR 2019 oral. If you use this code in your research, please cite the paper.


# Installation

***
```
pip install -r requirements.txt
```

# Preparing data
***
Edit `config.data_tr` in `config.py` to prepare data for different datasets.

```
python ./dump_data.py
```
# Training
***
For the first time running 'main.py' in each dataset, set the parameter `initialize` in `Data_Loader` to be `True`.

```
python ./main.py COLMAP #NARROW WIDE
```
