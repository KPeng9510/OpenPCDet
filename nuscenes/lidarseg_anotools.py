import sys
import glob
import os
import json
import numpy as np
import torch
filepath="/mrtstorage/users/kpeng/nu_lidar_seg/lidarseg/lidarseg/v1.0-trainval"
savepath="/mrtstorage/users/kpeng/nu_lidar_seg/processed_anno/"
"""
'car:5',    1
'truck:5', 2
 'construction_vehicle:5',5
 'bus:5', 3,4
 'trailer:5', 9

'barrier:5',21
 'motorcycle:5', 6
'bicycle:5',7
 'pedestrian:5',12,13,14,18
 'traffic_cone:5',20
flat 

"""
def eachFile(filepath):
    pathDir =  os.listdir(filepath)

    if not os.path.exists(savepath):
        os.mkdir(savepath)

    dic = {}
    for file_name in pathDir:
        file_path = os.path.join('%s/%s' % (filepath, file_name))
        save_file_path = os.path.join('%s/%s' % (savepath, file_name))
        semantic_labels = torch.from_numpy(np.fromfile(file_path, dtype=np.uint8,count=-1))
        points_number = len(semantic_labels)
        new_label = torch.zeros([points_number], dtype=semantic_labels.dtype, device=semantic_labels.device)

        mask_car = semantic_labels == 1
        mask_truck = semantic_labels == 2
        mask_c_vehicle = semantic_labels == 5
        mask_bus = (semantic_labels == 3) | (semantic_labels == 4)
        mask_trailer = semantic_labels == 9
        mask_barrier = semantic_labels == 21
        mask_motorcycle = semantic_labels == 6
        mask_bicycle = semantic_labels == 7
        mask_pedestrian = (semantic_labels == 12) | (semantic_labels == 13) | (semantic_labels == 14) | (semantic_labels == 18)
        mask_traffic_cone = semantic_labels == 20
        mask_driveable_area = semantic_labels == 24
        mask_flat_other = semantic_labels==27
        mask_sidewalk = semantic_labels == 25
        mask_terrain = semantic_labels ==26
        mask_others = semantic_labels == 27
        mask_static_others = (semantic_labels == 30)| (semantic_labels == 8)
        mask_vegetation = semantic_labels == 29
        mask_manmade = semantic_labels==28
        

        new_label[mask_car] = 1
        new_label[mask_truck] = 2
        new_label[mask_c_vehicle] = 3
        new_label[mask_bus] = 4
        new_label[mask_trailer] = 5
        new_label[mask_barrier] = 6
        new_label[mask_motorcycle] = 7
        new_label[mask_bicycle] = 8
        new_label[mask_pedestrian] = 9
        new_label[mask_traffic_cone] = 10
        new_label[mask_driveable_area] = 11
        new_label[mask_sidewalk] = 12
        new_label[mask_terrain] = 13
        new_label[mask_others] = 14
        new_label[mask_vegetation] = 15
        new_label[mask_manmade] = 16
        new_label[mask_flat_other] = 17
        new_label[mask_static_others]=18

        content= new_label.numpy().tobytes()

        #filepath='123.bin'
        binfile = open(save_file_path, 'wb+') #追加写入
        binfile.write(content)

        binfile.close()        

if __name__ == "__main__":
    eachFile(filepath)

