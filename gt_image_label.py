from collections import defaultdict
from pathlib import Path
import os
import pickle
import torch
import numpy as np
#import torch.utils.data as torch_data
#from data_loader_odo import data_loader
#from ..utils import common_utils
#from .augmentor.data_augmentor import DataAugmentor
#from .processor.data_processor import DataProcessor
#from .processor.point_feature_encoder import PointFeatureEncoder
import sys
from mapping import mapping
from voxelize import dense

def label_generator():
    file_name = "/home/kpeng/pc14/sample.pkl"
    save_path = "/mrtstorage/users/kpeng/kitti-semantics/label_image/"
    open_file = open(file_name, "rb")
    files_seq = pickle.load(open_file)
    open_file.close()
    for point_path in files_seq:
        dense_path = '/home/kpeng/pc14/kitti_odo/dense/' +str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1]
        dense_gt = np.fromfile(str(dense_path), dtype=np.float32, count=-1).reshape([-1, 5])[:, :5]
        pc_range = np.array([-50.0, -25.0, -5.0, 50.0, 25.0, 3.0],dtype=np.float32)
        voxel_size = np.array([0.1,0.1,8],dtype=np.float32)
        dense_gt[:,-1] = np.clip(dense_gt[:,-1],0,12)
        dense_gt = dense.compute_dense_gt(dense_gt, pc_range,voxel_size,13).reshape(1,13,500,1000)
        dense_gt = torch.from_numpy(dense_gt).permute(0,2,3,1).reshape(1*500*1000,13) # 2, 20, 500, 1000
        #print(dense_gt[:1,:])

        #sys.exit()
        weight_5_class = [1,2,3,4]
        weight_0_class = [0]
        weight_1_class = [5,6,7,8,9,10,11,12]
        dense_gt[:,weight_5_class]= dense_gt[:,weight_5_class]*5
        dense_gt[:,weight_0_class]=dense_gt[:,weight_0_class]*0
        dense_gt[:,weight_1_class]=dense_gt[:,weight_1_class]*1
        for i in range(12):
            dense_gt[:,i]+=0.12-0.01*i
            #print(debse_gt)
        dense_gt = torch.argmax(dense_gt,dim=-1).view(500*1000,1).numpy().astype(np.float32).tobytes()
        print(save_path+str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1])
        f=open(save_path+str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1],'wb')
        f.write(dense_gt)
        f.close()
        
if __name__ == '__main__':
    label_generator()




