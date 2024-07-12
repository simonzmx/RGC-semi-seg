import json
import os
import numpy as np
import scipy.io as scio
import skimage.io as skio
import scipy.ndimage as ndimage
import torch
import time
from argparse import Namespace
from typing import Dict
from pathlib import Path
from utils.post_processings import inference_whole_volume_segmentation, find_local_maximas
from utils.datasets import get_img_paths_by_mode
from utils.transforms import get_sphere_mask, insert_cell_sphere, normalize_volume, pad_volume, unpad_volume
from utils.measurements import calculate_average_precisions, measure_performance, measure_performance_pred_thresholded
from utils.visualizations import generate_precision_recall_curve, generate_model_prediction_map, generate_evaluation_figures
from utils.results import save_scores_to_csv


class Evaluator:
    def __init__(self,
                 args: Namespace,  # main args
                 hyp: Dict  # model config file
                 ):
        self.device = 'cuda' if not args.cpu and torch.cuda.is_available() else 'cpu'
        self.model_save_path = Path(hyp['model_save_path'])
        self.exp_name = args.exp_name if args.exp_name is not None else '.'.join(
            self.model_save_path.name.split('.')[:-1])
        self.results_path = os.path.join('runs', self.exp_name)
        if not os.path.exists(self.results_path):
            os.makedirs(self.results_path)

        self.num_classes = hyp['num_classes']

        # post-processing parameters
        self.local_maxima_filter = hyp['local_maxima_filter']
        self.median_filter = hyp['median_filter']
        self.dis_thresholds = hyp['dis_thresholds']
        self.voxel_res = hyp['voxel_resolution']

        # visualizations
        self.visualize_prob_map = args.visualize_prob_map

    def evaluate(self,
                 model,  # model to be evaluated
                 loss_sup,  # supervised loss
                 data_cfg,  # data configuration
                 mode,  # "val" or "test"
                 tfb_writer,  # tensorboard writer
                 epoch=0,  # training epoch
                 model_index=None,  # model index in CPS, otherwise None
                 train_info=None  # record for best scores
                 ):
        model.eval()
        img_paths = get_img_paths_by_mode(data_cfg, mode)
        sphere_mask = get_sphere_mask(data_cfg['cell_radius'], data_cfg['voxel_resolution'])
        scores = {'losses': [], 'avg_pres': [], 'pres': [], 'recs': [], 'f1s': [],
                  'pres_vt': [], 'recs_vt': [], 'f1s_vt': [], 'best_thresholds': []}
        all_precisions = []
        all_recalls = []
        with open(data_cfg['layer_seg_info'], 'r') as f:
            layer_info = json.loads(f.read())

        for img_path in img_paths:
            start_time = time.time()
            if mode == 'test':
                print(f'Testing on image {img_path}')
            img_filename = img_path.split(os.sep)[-1][:-4]
            this_dataset, this_subject, this_location = img_filename.split('_')[0: 3]

            # load this volume
            volume_ori = skio.imread(img_path).astype(np.float64)  # volume axis: y, z, x
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

            # run model inference on the whole volume
            if mode == 'test':
                if model_index is None:
                    pred_file = os.path.join(self.results_path, 'predictions', img_filename + '_pred.pt')
                else:
                    pred_file = os.path.join(self.results_path, f'predictions_{model_index}', img_filename + '_pred.pt')
                if os.path.exists(pred_file):
                    # load from disk
                    print(f'Loading predictions from disk for {img_path}...')
                    probs = torch.load(pred_file)
                    logits = None
                else:
                    # inference with the model
                    print(f'Running inference using trained model on {img_path}...')
                    if not os.path.exists(os.path.abspath(os.path.join(pred_file, os.pardir))):
                        os.makedirs(os.path.abspath(os.path.join(pred_file, os.pardir)))
                    this_mean = data_cfg['normalizations'][0] if data_cfg['normalizations'][0] is not None else np.mean(volume)
                    this_std = data_cfg['normalizations'][1] if data_cfg['normalizations'][1] is not None else np.std(volume)
                    volume, ori_size, _, _ = pad_volume(volume, data_cfg['window_size'])
                    volume = normalize_volume(volume, mean=this_mean, std=this_std)
                    logits, probs = inference_whole_volume_segmentation(model, volume, data_cfg, self.num_classes,
                                                                        self.device)  # tensor dimensions: classes, z, y, x
                    logits = unpad_volume(logits, ori_size)
                    probs = unpad_volume(probs, ori_size)
                    volume = unpad_volume(volume, ori_size)
                    torch.save(probs, pred_file)
            else:
                # inference with the model
                print(f'Running evaluation on {img_path}...')
                this_mean = data_cfg['normalizations'][0] if data_cfg['normalizations'][0] is not None else np.mean(volume)
                this_std = data_cfg['normalizations'][1] if data_cfg['normalizations'][1] is not None else np.std(volume)
                volume, ori_size, _, _ = pad_volume(volume, data_cfg['window_size'])
                volume = normalize_volume(volume, mean=this_mean, std=this_std)
                logits, probs = inference_whole_volume_segmentation(model, volume, data_cfg, self.num_classes,
                                                                    self.device)  # tensor dimensions: classes, z, y, x
                logits = unpad_volume(logits, ori_size)
                probs = unpad_volume(probs, ori_size)
                volume = unpad_volume(volume, ori_size)

            # calculate loss
            if logits is not None:
                this_loss = loss_sup(torch.unsqueeze(logits, 0).to(self.device),
                                    torch.unsqueeze(label, 0).to(self.device))
                scores['losses'].append(this_loss.item())

            # take ganglion cell map only
            preds_gc = probs[1].numpy()
            # apply median filter to remove spurious maxima
            preds_md = ndimage.median_filter(preds_gc, self.median_filter)
            # find local maximas
            preds_lmax = find_local_maximas(preds_md, self.local_maxima_filter)

            if mode == 'test' and self.visualize_prob_map:
                img_folder = img_filename if model_index is None else f'{img_filename}_{model_index}'
                generate_model_prediction_map(volume, label[1], preds_gc, self.results_path, img_folder,
                                              median_vol=preds_md, max_vol=preds_lmax)

            # measure model performance / select best threshold
            pres, recs, f1s, best_index, best_threshold = \
                measure_performance(preds_lmax, gt_crop_coords,
                                    volume.shape, self.voxel_res,
                                    dis_threshold=self.dis_thresholds[this_location],
                                    mode=mode)

            scores['pres'].append(pres[best_index])
            scores['recs'].append(recs[best_index])
            scores['f1s'].append(f1s[best_index])
            scores['avg_pres'].append(calculate_average_precisions(pres, recs))
            scores['best_thresholds'].append(best_threshold)
            all_precisions.append(pres)
            all_recalls.append(recs)

            if mode == 'test':
                # try to load training info from saved json file
                if train_info is None and os.path.exists(os.path.join(self.results_path, 'train_info.json')):
                    with open(os.path.join(self.results_path, 'train_info.json'), 'r') as rf:
                        train_info = json.loads(rf.read())
                # use best threshold from validation
                if train_info is not None and 'threshold' in train_info:
                    best_threshold = train_info['threshold']
                    print('Threshold selected based on the validation data: {:.4f}'.format(best_threshold))
                else:
                    print('Threshold selected based on the test data: {:.4f}'.format(best_threshold))

                # apply threshold
                preds_lmax_copy = np.copy(preds_lmax)
                preds_lmax_copy[preds_lmax_copy < best_threshold] = 0
                pred_coords = np.array(np.nonzero(preds_lmax_copy)).transpose()
                best_pre, best_rec, best_f1, gt_matched, pred_matched = \
                    measure_performance_pred_thresholded(pred_coords, gt_crop_coords, volume.shape,
                                                         self.voxel_res, self.dis_thresholds[this_location])
                scores['pres_vt'].append(best_pre)
                scores['recs_vt'].append(best_rec)
                scores['f1s_vt'].append(best_f1)
                # generate evaluation figures
                img_folder = img_filename if model_index is None else f'{img_filename}_{model_index}'
                img_folder += '_eval'
                generate_evaluation_figures(volume, pred_coords, gt_crop_coords, gt_matched, pred_matched, scores,
                                            self.results_path, img_folder, self.voxel_res)
            print("Testing on image {} took {:.4f} seconds.".format(img_path, time.time() - start_time))

        eval_loss = np.mean(scores['losses'])
        eval_ap = np.mean(scores['avg_pres'])
        mean_precisions = np.mean(np.array(all_precisions), axis=0)
        mean_recalls = np.mean(np.array(all_recalls), axis=0)

        # logging
        if model_index is None:
            tfb_writer.add_scalar(f'Loss/{mode}', eval_loss, epoch)
            tfb_writer.add_scalar(f'AP/{mode}', eval_ap, epoch)
            tfb_writer.add_figure(f'precision-recall curve epoch={epoch}/{mode}',
                                  generate_precision_recall_curve(mean_precisions, mean_recalls))
        else:
            tfb_writer.add_scalar(f'Loss_{model_index}/{mode}', eval_loss, epoch)
            tfb_writer.add_scalar(f'AP_{model_index}/{mode}', eval_ap, epoch)
            tfb_writer.add_figure(f'precision-recall curve epoch={epoch}_{model_index}/{mode}',
                                  generate_precision_recall_curve(mean_precisions, mean_recalls))
        print(
            '{}: loss={:.4f} average_precision={:.4f}({:.4f}) pre={:.4f}({:.4f}) rec={:.4f}({:.4f}) f1={:.4f}({:.4f}).'.format(
                mode, eval_loss,
                np.mean(scores['avg_pres']), np.std(scores['avg_pres']),
                np.mean(scores['pres']), np.std(scores['pres']),
                np.mean(scores['recs']), np.std(scores['recs']),
                np.mean(scores['f1s']), np.std(scores['f1s'])
            ))

        if mode == 'test':
            save_scores_to_csv(scores, img_paths, self.results_path, model_index=model_index, train_info=train_info)

        return eval_loss, eval_ap, np.mean(scores['f1s']), scores['best_thresholds'][0]
