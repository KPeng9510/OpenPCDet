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
#from mapping import mapping
from voxelize.voxelize import dense
color_map={
  "0" : [255, 255, 255],
  "1": [0, 0, 255],
  "11": [255, 120, 60],
  "3": [255, 255, 0],
  "4": [250, 80, 100],
  "5": [255, 0, 255],
  "8": [245, 230, 100],
  "10": [0, 175, 0],
  "6": [75, 0, 75],
  "7": [150, 60, 30],
  "2": [180, 30, 80],
  "12": [150, 240, 80],
  "9": [135,60,0]
}
def recursive_glob(rootdir=".", suffix=""):
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)
        for filename in filenames
        if filename.endswith(suffix)
    ]

def colorized_image_generator(files_seq):
    #file_name = "/home/kpeng/pc14/sample_test.pkl"
    save_path = "/mrtstorage/users/kpeng/nu_lidar_seg/colorized_gt/"
    open_file = open(file_name, "rb")
    files_seq = pickle.load(open_file)
    open_file.close()
    #print(files_seq)
    #sys.exit()
    #locate = "/home/kpeng/pc14/kitti_odo/training/08/0000168.bin"
    #index = files_seq.index(locate)
    for point_path in files_seq:
        dense_path = '/mrtstorage/users/kpeng/nu_lidar_seg/label_image_1/' +str(point_path).split('/')[-1]
        pointcloud = np.fromfile(str(dense_path), dtype=np.int8, count=-1).reshape([500,1000, 1])
        picture = np.zeros([500,1000,3])
        for i in range(0,16):
            mask = pointcloud[:,:,-1] == i
            #print(mask.shape)
            if mask.sum() == 0:
                continue
            picture[mask]=np.array(color_map[str(i)])/255
            #plt.figure()
            #picture=np.transpose()
        img_path = save_path +str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1].split('.')[0]+'.png'
        print(img_path)
        plt.imsave(img_path,picture)

def label_generator():
    #file_name = "/home/kpeng/pc14/sample_test.pkl"
    save_path = "/mrtstorage/users/kpeng/nu_lidar_seg/label_bin/"
    #open_file = open(file_name, "rb")
    file_path = "/mrtstorage/users/kpeng/nu_lidar_seg/concat_lidar_flat_divided/new_2/samples/LIDAR_TOP/"
    #files_seq = pickle.load(open_file)
    files_seq = recursive_glob(rootdir=file_path, suffix=".bin")
    #open_file.close()
    #locate = "/home/kpeng/pc14/kitti_odo/training/10/000362.bin"
    #index = files_seq.index(locate)
    for point_path in files_seq:
        dense_path = point_path
        dense_gt = np.fromfile(str(dense_path), dtype=np.float32, count=-1).reshape([-1, 6])[:, :6]
        pc_range = np.array([-51.2, -51.2, -5.0, 51.2, 51.2, 3.0],dtype=np.float32)
        voxel_size = np.array([0.2,0.2,8],dtype=np.float32)
        dense_gt[:,-1] = np.clip(dense_gt[:,-1],0,19)
        dense_gt = dense.compute_dense_gt(dense_gt, pc_range,voxel_size,20).reshape(1,20,512,512)
        dense_gt = torch.from_numpy(dense_gt).permute(0,2,3,1).reshape(1*512*512,20) # 2, 20, 500, 1000
        #print(dense_gt[:1,:])

        #sys.exit()
        weight_5_class = [1,2,3,4,7,8,9]
        weight_0_class = [0]
        weight_1_class = [5,6,10,11,12,13,14,15,16,17,18,19]
        dense_gt[:,weight_5_class]= dense_gt[:,weight_5_class]*5
        dense_gt[:,weight_0_class]=dense_gt[:,weight_0_class]*0
        dense_gt[:,weight_1_class]=dense_gt[:,weight_1_class]*1
        for i in range(20):
            dense_gt[:,i]+=0.12-0.01*i
            #print(debse_gt)
        dense_gt = torch.argmax(dense_gt,dim=-1).view(512*512,1).numpy().astype(np.float32).tobytes()
        print(save_path+str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1])
        f=open(save_path+str(point_path).split('/')[-2] +'/'+ str(point_path).split('/')[-1],'wb')
        f.write(dense_gt)
        f.close()

if __name__ == '__main__':
    label_generator()




