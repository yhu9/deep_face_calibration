import itertools
import argparse
import os

from numpy.core.records import fromarrays
import scipy.io
import torch
import numpy as np

#from model2 import PointNet
#import dataloader
#import util
#import time



base_dir = "/home/huynshen/data/tmp/monocular_depth_estimation/results"
biwi_dir = ""
video_mask_dir = "/home/huynshen/projects/git/EPnP/matlab/preprocess"

results_dir = '/home/huynshen/projects/deep_face_calibration/source/results/test6'
biwi_x = 'pointnetbn_opt_biwi.mat'
biwiid_x = 'pointnetbn_opt_biwiid.mat'
cad120_x = 'pointnetbn_opt_cad120.mat'
human36_x = 'pointnetbn_human36.mat'

biwi_dir = ''

dir_list = os.listdir(base_dir)
print(dir_list)
quit()

print(base_dir)



