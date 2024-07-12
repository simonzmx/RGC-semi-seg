import os
import json
import time
import torch
import yaml
import numpy as np
import skimage.io as skio
import scipy.io as scio
import scipy.ndimage as ndimage
from utils.transforms import get_sphere_mask, insert_cell_sphere, normalize_volume, pad_volume, unpad_volume
from utils.post_processings import inference_whole_volume_segmentation, find_local_maximas
from utils.measurements import calculate_average_precisions, measure_performance, measure_performance_pred_thresholded
from utils.visualizations import generate_precision_recall_curve, generate_model_prediction_map, generate_evaluation_figures
from utils.results import save_scores_to_csv


data_cfg_file = './cfg/data/dataset_iu.yaml'
model_cfg_file = './cfg/model/hyp.semi_seg_cps_iu.yaml'
# Load data settings from the yaml file
with open(data_cfg_file, 'rb') as f:
    data_cfg = yaml.load(f, Loader=yaml.SafeLoader)
# Load hyper-parameters from the yaml file
with open(model_cfg_file, 'rb') as f:
    hyp = yaml.load(f, Loader=yaml.SafeLoader)

with open(data_cfg['layer_seg_info'], 'r') as f:
    layer_info = json.loads(f.read())
sphere_mask = get_sphere_mask(data_cfg['cell_radius'], data_cfg['voxel_resolution'])


def ensemble_predictions(pred_list, method, training_infos):
    if method == 'mean':
        return sum(pred_list) / len(pred_list)
    elif method == 'mean_thresholded':
        assert len(pred_list) == len(training_infos)
        for idx in range(len(pred_list)):
            this_threshold = training_infos[idx]['threshold']
            this_pred = pred_list[idx]
            this_pred = (this_pred - this_threshold) / (this_pred.max() - this_threshold)
            this_pred[this_pred < 0] = 0
            pred_list[idx] = this_pred
        return sum(pred_list) / len(pred_list)
    else:
        raise ValueError('Unsupported ensemble method.')


def ensemble_test(pred_dirs, target_dir, method):
    # load training info
    training_infos = []
    for pred_dir in pred_dirs:
        parent_dir = os.path.abspath(os.path.join(pred_dir, os.pardir))
        with open(os.path.join(parent_dir, 'train_info.json'), 'r') as rf:
            train_info = json.loads(rf.read())
        if pred_dir.endswith('_1'):
            training_infos.append(train_info['model_1'])
        elif pred_dir.endswith('_2'):
            training_infos.append(train_info['model_2'])
        else:
            training_infos.append(train_info)

    scores = {'losses': [], 'avg_pres': [], 'pres': [], 'recs': [], 'f1s': [],
              'pres_vt': [], 'recs_vt': [], 'f1s_vt': [], 'best_thresholds': []}
    all_precisions = []
    all_recalls = []
    img_paths = []
    for pred_file in os.listdir(pred_dirs[0]):
        pred_list = [torch.load(os.path.join(pred_dirs[0], pred_file))]
        for dir_idx in range(1, len(pred_dirs)):
            this_pred_file = os.path.join(pred_dirs[dir_idx], pred_file)
            if not os.path.exists(this_pred_file):
                raise FileNotFoundError(f'Prediction file not exist: {this_pred_file}')
            this_probs = torch.load(this_pred_file)
            pred_list.append(this_probs)

        start_time = time.time()
        filename = pred_file.split(os.sep)[-1][:-3]
        this_dataset, this_subject, this_location = filename.split('_')[0: 3]

        # find the input volume
        img_filename = None
        for imgf in os.listdir(os.path.join(data_cfg['root_dir'], 'images')):
            if imgf.startswith('_'.join([this_dataset, this_subject, this_location])):
                img_paths.append(imgf)
                img_filename = imgf.split(os.sep)[-1][:-4]
                break
        if img_filename is None:
            raise FileNotFoundError(f'Image not found for: {filename}')

        # load this volume
        volume_ori = skio.imread(os.path.join(data_cfg['root_dir'], 'images', img_filename + '.tif')).astype(np.float64)  # volume axis: y, z, x
        volume_ori = np.transpose(volume_ori, axes=(1, 0, 2))  # volume axis: z, y, x

        # load label coordinates
        label_file_path = os.path.join(data_cfg['root_dir'], 'labels', img_filename + '_marker.mat')
        mat_content = scio.loadmat(label_file_path)
        gt_coords = mat_content['temp_marker'].astype(np.uint32)  # coordinates axis: z, x, y
        gt_coords[:, [1, 2]] = gt_coords[:, [2, 1]]  # coordinates axis: z, y, x
        gt_coords -= 1  # original coordinates start from 1, make it start from 0

        # crop (z-axis) at the retinal layer region
        # TODO: Ganglion Cell Layer segmentation according to WeakGCSeg paper, now using information provided by the authors
        if img_filename in layer_info:
            min_z = max(layer_info[img_filename]['min_z'], 0)
            max_z = min(layer_info[img_filename]['max_z'] + 1, volume_ori.shape[0])
        else:
            min_z = max(np.min(gt_coords[:, 0]), 0)
            max_z = min(np.max(gt_coords[:, 0]) + 1, volume_ori.shape[0])
        volume = volume_ori[min_z: max_z, :, :]
        gt_crop_coords = gt_coords - np.array([min_z, 0, 0])  # convert gc coordinates to the cropped volume

        # build label volume
        label = np.zeros(volume.shape).astype(np.uint8)
        for coord_idx in range(gt_crop_coords.shape[0]):
            label = insert_cell_sphere(label, gt_crop_coords[coord_idx, :], sphere_mask)
        label = torch.tensor(np.array([1 - label, label]), dtype=torch.float32)  # tensor dimensions: classes, z, y, x

        # apply ensemble
        probs = ensemble_predictions(pred_list, method, training_infos)

        # take ganglion cell map only
        preds_gc = probs[1].numpy()
        # apply median filter to remove spurious maxima
        preds_md = ndimage.median_filter(preds_gc, hyp['median_filter'])
        # find local maximas
        preds_lmax = find_local_maximas(preds_md, hyp['local_maxima_filter'])

        # measure model performance / select best threshold
        pres, recs, f1s, best_index, best_threshold = \
            measure_performance(preds_lmax, gt_crop_coords,
                                volume.shape, hyp['voxel_resolution'],
                                dis_threshold=hyp['dis_thresholds'][this_location],
                                mode='test')

        scores['pres'].append(pres[best_index])
        scores['recs'].append(recs[best_index])
        scores['f1s'].append(f1s[best_index])
        scores['avg_pres'].append(calculate_average_precisions(pres, recs))
        scores['best_thresholds'].append(best_threshold)
        all_precisions.append(pres)
        all_recalls.append(recs)

        # apply threshold
        preds_lmax_copy = np.copy(preds_lmax)
        preds_lmax_copy[preds_lmax_copy < best_threshold] = 0
        pred_coords = np.array(np.nonzero(preds_lmax_copy)).transpose()
        best_pre, best_rec, best_f1, gt_matched, pred_matched = \
            measure_performance_pred_thresholded(pred_coords, gt_crop_coords, volume.shape,
                                                 hyp['voxel_resolution'], hyp['dis_thresholds'][this_location])
        scores['pres_vt'].append(best_pre)
        scores['recs_vt'].append(best_rec)
        scores['f1s_vt'].append(best_f1)
        # generate evaluation figures
        img_folder = f'{img_filename}_eval'
        generate_evaluation_figures(volume, pred_coords, gt_crop_coords, gt_matched, pred_matched, scores,
                                    target_dir, img_folder, hyp['voxel_resolution'])
        print("Testing on image {} took {:.4f} seconds.".format(filename, time.time() - start_time))

    eval_loss = np.mean(scores['losses'])
    eval_ap = np.mean(scores['avg_pres'])
    mean_precisions = np.mean(np.array(all_precisions), axis=0)
    mean_recalls = np.mean(np.array(all_recalls), axis=0)

    print(
        'test: loss={:.4f} average_precision={:.4f}({:.4f}) pre={:.4f}({:.4f}) rec={:.4f}({:.4f}) f1={:.4f}({:.4f}).'.format(
            eval_loss,
            np.mean(scores['avg_pres']), np.std(scores['avg_pres']),
            np.mean(scores['pres']), np.std(scores['pres']),
            np.mean(scores['recs']), np.std(scores['recs']),
            np.mean(scores['f1s']), np.std(scores['f1s'])
        ))

    save_scores_to_csv(scores, img_paths, target_dir, model_index=None, train_info=None)

    return eval_loss, eval_ap, np.mean(scores['f1s']), scores['best_thresholds'][0]


if __name__ == '__main__':
    label_num = 1
    subject = '053'
    method = 'mean'
    pred_dirs = [
        f'./runs/semi_seg_cps_label{label_num}/semi_seg_cps_m1_label{label_num}_test{subject}/predictions_1',
        f'./runs/semi_seg_cps_label{label_num}/semi_seg_cps_m1_label{label_num}_test{subject}/predictions_2',
        # f'./runs/semi_seg_cct_label{label_num}/semi_seg_cct_label{label_num}_test{subject}/predictions'
    ]
    target_dir = f'./runs/archives/semi_seg_cps_label{label_num}/semi_seg_cps_m1_label{label_num}_test{subject}_ensemble_{method}'
    ensemble_test(pred_dirs, target_dir, method)
