import pickle
import time
#import cv2
from PIL import Image
import numpy as np
import torch
import tqdm

from pcdet.models import load_data_to_gpu
from pcdet.utils import common_utils

color_map = {
    "0": [255, 255, 255],  #
    "1": [0, 0, 255],  # vehicle
    "2": [ 0, 0, 70], #truck
    "3": [0, 80,100], #construction_vehicle
    "4": [0,  0, 90], #bus
    "5": [ 0,  0,110], # trailer
    "6": [255, 120, 50], # barrier
    "9": [255, 30, 30],  # person
    "7": [30, 60, 150],  # motorcycle
    "8": [119, 11, 32], #bicycle
    "11": [255, 0, 255],  # road
    "12": [75, 0, 75],  # sidewalk
    "17": [255, 150, 255],  # other ground
    "16": [255, 200, 0],  # building
    "10": [255, 120, 50],  # object
    "15": [0, 175, 0],  # vegetation
    "18": [135, 60, 0],  # trunk
    "13": [150, 240, 80],  # terrain
    "14": [20,30,100]
}
def id_to_rgb(pred_id):
    shape = list(pred_id.shape)[:2]
    shape.append(3)
    rgb = np.zeros(shape, dtype=np.uint8) + 255
    for i in range(0, 18):
        mask = pred_id[:, :, 0] == i
        # print(mask.shape)
        if mask.sum() == 0:
            continue
        rgb[mask] = np.array(color_map[str(i + 1)])
    return rgb.astype(np.uint8)


def statistics_info(cfg, ret_dict, metric, disp_dict):
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] += ret_dict.get('roi_%s' % str(cur_thresh), 0)
        metric['recall_rcnn_%s' % str(cur_thresh)] += ret_dict.get('rcnn_%s' % str(cur_thresh), 0)
    metric['gt_num'] += ret_dict.get('gt', 0)
    min_thresh = cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST[0]
    disp_dict['recall_%s' % str(min_thresh)] = \
        '(%d, %d) / %d' % (metric['recall_roi_%s' % str(min_thresh)], metric['recall_rcnn_%s' % str(min_thresh)], metric['gt_num'])
def get_iou(pred, gt, intersect, union, n_classes=18):
    for i in range(len(pred)):
        pred_tmp = pred[i]
        gt_tmp = gt[i]
        for j in range(n_classes):
            match = (pred_tmp == j).int() + (gt_tmp == j).int()
            it = torch.sum(match == 2).item()
            un = torch.sum(match > 0).item()
            intersect[j] += it
            union[j] += un
    return intersect, union

def eval_one_epoch(cfg, model, dataloader, epoch_id, logger, dist_test=False, save_to_file=False, result_dir=None):
    result_dir.mkdir(parents=True, exist_ok=True)
    save_dir_att = result_dir/'segmentation_eval'/'att'
    save_dir_att.mkdir(parents=True, exist_ok=True)
    save_dir_pred = result_dir/'segmentation_eval'/'pred'
    save_dir_pred.mkdir(parents=True, exist_ok=True)
    save_dir_gt = result_dir/'segmentation_eval'/'gt'
    save_dir_gt.mkdir(parents=True, exist_ok=True)

    final_output_dir = result_dir / 'final_result' / 'data'
    if save_to_file:
        final_output_dir.mkdir(parents=True, exist_ok=True)

    metric = {
        'gt_num': 0,
    }
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        metric['recall_roi_%s' % str(cur_thresh)] = 0
        metric['recall_rcnn_%s' % str(cur_thresh)] = 0

    dataset = dataloader.dataset
    class_names = dataset.class_names
    det_annos = []

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

    if cfg.LOCAL_RANK == 0:
        progress_bar = tqdm.tqdm(total=len(dataloader), leave=True, desc='eval', dynamic_ncols=True)
    start_time = time.time()
    inter = torch.zeros(18)
    union = torch.zeros(18)
    for i, batch_dict in enumerate(dataloader):
        load_data_to_gpu(batch_dict)
        with torch.no_grad():
            pred_dicts, ret_dict, data_dict = model(batch_dict)
        disp_dict = {}
        pred = data_dict["prediction"]
        gt = data_dict["labels_seg"]
        att = data_dict["attention"]
        z_mask = gt==0
        gt-=1
        pred[z_mask]=-1
        inter, union = get_iou(pred, gt, inter, union)
        statistics_info(cfg, ret_dict, metric, disp_dict)
        annos = dataset.generate_prediction_dicts(
            batch_dict, pred_dicts, class_names,
            output_path=final_output_dir if save_to_file else None
        )
        det_annos += annos
        if cfg.LOCAL_RANK == 0:
            progress_bar.set_postfix(disp_dict)
            progress_bar.update()
        batch_size=data_dict["batch_size"]
        for batch_index in range(batch_size):
            prediction = id_to_rgb(pred[batch_index].permute(1,2,0).cpu())
            image_save_dir = save_dir_pred/('%06d.png' % int(i * batch_size + batch_index))
            prediction= Image.fromarray(prediction)
            prediction.save(image_save_dir)
            gtt = id_to_rgb((gt[batch_index].permute(1,2,0)-1).cpu())
            gt_save_dir = save_dir_gt/('%06d.png' % int(i * batch_size + batch_index))
            gtt = Image.fromarray(gtt[batch_index])
            gtt.save(gt_save_dir)
        #break
    class_list = []
    iou = []

    class_name = ["car","truck","c_vehicle","bus","trailer","barrier","motorcycle","bicycle","pedestrain","traffic_cone","driveable_area","sidewalk","terrain","others","vegetation","manmade","flat_others","static_objects"]
    for k in range(len(inter)):
        if union[k] != 0:
            class_list.append(class_name[k])
            iou.append(inter[k] / union[k])
        else:
            continue
    miou = ((sum(iou)) / (len(iou)))

    print("*******************************************************")
    for j, class_index in enumerate(class_name):
        #print(j)
        print('iou_%s: %f' % (class_index, iou[j]))
        #ret_dict['iou_%s' % class_index] = iou[j]
    print('miou: %f' % (miou) + '\n')
    #ret_dict['miou'] = miou
    #file.write("****************************************************\n")
    print("****************************************************")


    if cfg.LOCAL_RANK == 0:
        progress_bar.close()

    if dist_test:
        rank, world_size = common_utils.get_dist_info()
        det_annos = common_utils.merge_results_dist(det_annos, len(dataset), tmpdir=result_dir / 'tmpdir')
        metric = common_utils.merge_results_dist([metric], world_size, tmpdir=result_dir / 'tmpdir')

    logger.info('*************** Performance of EPOCH %s *****************' % epoch_id)
    sec_per_example = (time.time() - start_time) / len(dataloader.dataset)
    logger.info('Generate label finished(sec_per_example: %.4f second).' % sec_per_example)

    if cfg.LOCAL_RANK != 0:
        return {}

    ret_dict = {}
    if dist_test:
        for key, val in metric[0].items():
            for k in range(1, world_size):
                metric[0][key] += metric[k][key]
        metric = metric[0]

    gt_num_cnt = metric['gt_num']
    for cur_thresh in cfg.MODEL.POST_PROCESSING.RECALL_THRESH_LIST:
        cur_roi_recall = metric['recall_roi_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        cur_rcnn_recall = metric['recall_rcnn_%s' % str(cur_thresh)] / max(gt_num_cnt, 1)
        logger.info('recall_roi_%s: %f' % (cur_thresh, cur_roi_recall))
        logger.info('recall_rcnn_%s: %f' % (cur_thresh, cur_rcnn_recall))
        ret_dict['recall/roi_%s' % str(cur_thresh)] = cur_roi_recall
        ret_dict['recall/rcnn_%s' % str(cur_thresh)] = cur_rcnn_recall

    total_pred_objects = 0
    for anno in det_annos:
        total_pred_objects += anno['name'].__len__()
    logger.info('Average predicted number of objects(%d samples): %.3f'
                % (len(det_annos), total_pred_objects / max(1, len(det_annos))))

    with open(result_dir / 'result.pkl', 'wb') as f:
        pickle.dump(det_annos, f)

    result_str, result_dict = dataset.evaluation(
        det_annos, class_names,
        eval_metric=cfg.MODEL.POST_PROCESSING.EVAL_METRIC,
        output_path=final_output_dir
    )

    logger.info(result_str)
    ret_dict.update(result_dict)

    logger.info('Result is save to %s' % result_dir)
    logger.info('****************Evaluation done.*****************')
    return ret_dict


if __name__ == '__main__':
    pass
