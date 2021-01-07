import pickle
import time
import math
import numpy as np
import torch
import tqdm
#import Path
from pathlib import Path
from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils
import sys
import time
import os
def get_iou(pred, gt,number,intersect,union,index,logger,result_dir,time_stamp, n_classes=12):
    total_miou = 0.0
    class_name=["vehicle","person","two-wheel","rider","road","sidewalk","otherground","building","object","vegetation","trunk","terrain"]
    iou_list_sum = torch.zeros([n_classes])
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]
        for j in range(n_classes):
            match = (pred_tmp == j).int() + (gt_tmp == j).int()
            
            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()
            #it_list[j]+=it
            #un_list[]
            intersect[j] += it
            union[j] += un

        iou = []
        #unique_label = #np.unique(gt_tmp.data.cpu().numpy())
        zero_count=0
        class_list=[]
    
    #file = open(result_dir+"/log_eval_step_50_"+timestamp+".txt", 'a')

    if (number % 50)|(number ==index) == 0:
        file = open(result_dir/("eval_"+time_stamp+".txt"), 'a')
        for k in range(len(intersect)):
            if union[k]!=0:
                class_list.append(class_name[k])
                iou.append(intersect[k] / union[k])
            else:
                continue;
        miou = ((sum(iou)) / (len(iou)))
        iou = torch.Tensor(iou)
        file.write("******************eval_%f******************"%number)
        print("*******************eval_result**********************:")
        for j, class_index in enumerate(class_list):
            logger.info('iou_%s: %f' % (class_index, iou[j]))
            file.write('iou_%s: %f' % (class_index, iou[j]))
        logger.info('miou: %f' % (miou))
        file.write('miou: %f' % (miou))
        file.write("****************************************************")
        print("****************************************************")
        #file.write("******************eval_%f******************"%number)
        #file.write()
        file.close()
    return intersect, union

def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])


def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    t = time.localtime()
    timestamp = time.strftime('%b-%d-%Y_%H%M', t)
    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info('*************** EPOCH %s EVALUATION *****************' % epoch_id)
    if dist_test:
        num_gpus = torch.cuda.device_count()
        local_rank = cfg.LOCAL_RANK % num_gpus
        model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[local_rank],
                broadcast_buffers=False
        )
    model.eval()
    start_time = time.time()
    n_val=len(dataloader)
    with tqdm.tqdm(total=n_val, desc='Validation round', unit='batch', leave=False) as pbar:
        intersection = torch.zeros(12)
        union = torch.zeros(12)
        for i, batch_dict in enumerate(dataloader):
            
            load_data_to_gpu(batch_dict)
            
            with torch.no_grad():
                pred_dict= model(batch_dict)
            torch.set_printoptions(profile="full")
            batch_size,c,h,w = pred_dict["prediction"].size()
            dense_gt = pred_dict["labels_seg"] # 0 to 12
            observation = pred_dict["observation"][:,:500,:1000].unsqueeze(1)
            no_obser_mask = observation <= 0
            nonzero_mask = dense_gt.view(batch_size,1,h,w) != 0
            mask = torch.zeros_like(observation) # 2,500,1000
            mask[nonzero_mask]=1
            mask[no_obser_mask]=0
            anti_mask = ~(mask.bool())
            dense_gt[anti_mask]=0
            pred = pred_dict["prediction"][:,:12,:,:]
            dense_gt = dense_gt -1 #from -1 to 11
            pred = torch.argmax(pred, dim=1,keepdim=True)
            prediction_save_path = result_dir/"eval_segmentation/"
            if os.path.exists(prediction_save_path) != True:
                os.mkdir(prediction_save_path)
            for batch_index in range(batch_size):
                
                image_save_path = prediction_save_path/(str(i*batch_size+batch_index)+".bin")
                #if os.path.exists(image_save_path):
                #    os.mkdir(image_save_path)

                label = pred[batch_index].flatten().cpu().numpy().astype(np.float32).tobytes()
                f=open(image_save_path,'wb+')
                f.write(label)
                f.close()
            #pred = pred.permute(0,2,3,1)
            pred[anti_mask] = -1
            #intersection = torch.zeros(12)
            #union = torch.zeros(12)
            #pred = pred.permute(0,2,3,1)
            intersection,union= get_iou(pred,dense_gt,i+1,intersection,union, n_val*batch_size,logger,result_dir,timestamp)
            
            
            pbar.update()
            #if ((i+1)%50 == 0)&(i!=0):
            #    iou_middle = iou_out/((i+1)*batch_size)
            #    miou_middle = miou_out/((i+1)*batch_size)
            #    print("iou over %s samples:"%i)
            #    print(iou_middle-torch.floor(iou_middle))
            #    print("miou over %s samples:"%i)
            #    print(miou_middle)
        #iou_out = iou_out/(n_val*batch_size)
        #miou_out = miou_out/(n_val*batch_size)
        #class_name=["vehicle","person","two-wheel","rider","road","sidewalk","otherground","building","object","vegetation","trunk","terrain"]
        #for j, class_index in enumerate(class_name):
        #    logger.info('iou_%s: %f' % (class_index, iou_out[j]))
        #logger.info('miou: %f' % (miou_out))
        
    """
    logger.info('****************Evaluation done.*****************')
    return ret_dict
    """
if __name__ == '__main__':
    pass
