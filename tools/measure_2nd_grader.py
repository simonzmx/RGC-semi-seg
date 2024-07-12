import numpy as np
import os
import scipy.io as scio
import skimage.io as skio
import json


def build_gt2pred_ordered_mapping(gt_coords, pred_coords, voxel_res, dis_threshold):
    gt2pred_ordered_map = {}
    for gt_idx, gt_coord in enumerate(gt_coords):
        available_preds = []
        for pred_idx, pred_coord in enumerate(pred_coords):
            # calculate ground-truth to prediction distance
            distance = np.sqrt((voxel_res[0] * (gt_coord[0] - pred_coord[0])) ** 2 +
                               (voxel_res[1] * (gt_coord[1] - pred_coord[1])) ** 2 +
                               (voxel_res[2] * (gt_coord[2] - pred_coord[2])) ** 2)
            if distance < dis_threshold:
                available_preds.append((pred_idx,  # index
                                        1,  # probability
                                        distance))  # distance
        preds_sorted = sorted(available_preds, key=lambda x: x[2])  # sorted by distance
        gt2pred_ordered_map[gt_idx] = preds_sorted
    return gt2pred_ordered_map


def measure_performance(pred_coords, gt_coords, volume_shape, voxel_res, dis_threshold, border_size=10):
    gt2pred_ordered_map = build_gt2pred_ordered_mapping(gt_coords, pred_coords, voxel_res, dis_threshold)

    # disregard border voxels for ground-truth
    gt_matched_master = np.zeros(gt_coords.shape[0])
    for gt_idx, gt_coord in enumerate(gt_coords):
        if not (border_size <= gt_coord[0] < volume_shape[0] - border_size and
                border_size <= gt_coord[1] < volume_shape[1] - border_size and
                border_size <= gt_coord[2] < volume_shape[2] - border_size):
            gt_matched_master[gt_idx] = -1

    gt_matched = np.copy(gt_matched_master)

    pred_matched = np.zeros(pred_coords.shape[0])
    for pred_idx, pred_coord in enumerate(pred_coords):
        # discard border voxels for predicted coordinates
        if not (border_size <= pred_coord[0] < volume_shape[0] - border_size and
                border_size <= pred_coord[1] < volume_shape[1] - border_size and
                border_size <= pred_coord[2] < volume_shape[2] - border_size):
            pred_matched[pred_idx] = -1

    for gt_idx, gt_coord in enumerate(gt_coords):
        if gt_matched[gt_idx] == 0:
            for matched_pred in gt2pred_ordered_map[gt_idx]:
                if pred_matched[matched_pred[0]] == 0:  # this prediction is available
                    gt_matched[gt_idx] = matched_pred[0] + 1  # keep 0 indicating non-matched, thus +1
                    pred_matched[matched_pred[0]] = gt_idx + 1  # keep 0 indicating non-matched, thus +1
                    break

    # calculate precision and recall
    true_pos = np.count_nonzero(gt_matched > 0)
    false_pos = np.count_nonzero(pred_matched == 0)
    false_neg = np.count_nonzero(gt_matched == 0)
    pre = true_pos / (true_pos + false_pos + 1e-6)
    rec = true_pos / (true_pos + false_neg + 1e-6)
    f1 = 2 * pre * rec / (pre + rec + 1e-6)

    return pre, rec, f1


def main(data_dir, dataset='IU', locations=[], sbj_type='Healthy'):
    img_dir = os.path.join(data_dir, 'images')
    g1_dir = os.path.join(data_dir, 'labels')
    g2_dir = os.path.join(data_dir, '2nd grader')
    with open('D:\OSU\optometry\GC-Segmentation\src\data/all/gc_layers_all.json', 'r') as f:
        layer_info = json.loads(f.read())

    voxel_resolution = [0.94, 0.97, 0.97] if dataset == 'IU' else [0.685, 1.5, 1.5]
    precisions = []
    recalls = []
    f1s = []
    for img_path in os.listdir(img_dir):
        if img_path.endswith('.tif') and dataset in img_path:
            img_filename = img_path.split(os.sep)[-1][:-4]
            this_dataset, this_subject, this_location = img_filename.split('_')[0: 3]
            if this_location not in locations or sbj_type not in img_filename:
                continue
            print(img_filename)

            if dataset == 'IU':
                if this_location == '3T':
                    dis_threshold = 5.85  # IU 3T
                elif this_location == '8T':
                    dis_threshold = 7.81  # IU 8T
                else:
                    dis_threshold = 8.78  # Healthy 12T
            else:
                if 'Healthy' in img_path:
                    dis_threshold = 8.78  # Healthy 12T
                else:
                    dis_threshold = 10.78  # Glaucoma 12T

            # load this volume
            volume_ori = skio.imread(os.path.join(img_dir, img_path))  # volume axis: y, z, x
            volume_ori = np.transpose(volume_ori, axes=(1, 0, 2))  # volume axis: z, y, x

            # load g1 coordinates
            label_file_path1 = os.path.join(g1_dir, img_filename + '_marker.mat')
            mat_content1 = scio.loadmat(label_file_path1)
            coords1 = mat_content1['temp_marker'].astype(np.uint32)  # coordinates axis: z, x, y
            coords1[:, [1, 2]] = coords1[:, [2, 1]]  # coordinates axis: z, y, x
            coords1 -= 1  # original coordinates start from 1, make it start from 0

            # load g2 coordinates
            label_file_path2 = os.path.join(g2_dir, img_filename + '_marker.mat')
            mat_content2 = scio.loadmat(label_file_path2)
            coords2 = mat_content2['temp_marker'].astype(np.uint32)  # coordinates axis: z, x, y
            coords2[:, [1, 2]] = coords2[:, [2, 1]]  # coordinates axis: z, y, x
            coords2 -= 1  # original coordinates start from 1, make it start from 0

            # crop (z-axis) at the retinal layer region
            # TODO: do this step as the original paper does, not using the ground-truth
            if img_filename in layer_info:
                min_z = max(layer_info[img_filename]['min_z'], 0)
                max_z = min(layer_info[img_filename]['max_z'] + 1, volume_ori.shape[0])
            else:
                min_z = max(np.min(coords1[:, 0]), 0)
                max_z = min(np.max(coords1[:, 0]) + 1, volume_ori.shape[0])
            volume = volume_ori[min_z: max_z, :, :]
            coords1 = coords1 - np.array([min_z, 0, 0])  # convert gc coordinates to the cropped volume
            coords2 = coords2 - np.array([min_z, 0, 0])

            pre, rec, f1 = measure_performance(coords2, coords1, volume.shape, voxel_resolution, dis_threshold)
            precisions.append(pre)
            recalls.append(rec)
            f1s.append(f1)

    print("Precision Avg/Std = {} / {}".format(np.mean(precisions), np.std(precisions)))
    print("Recall Avg/Std = {} / {}".format(np.mean(recalls), np.std(recalls)))
    print("F1 Avg/Std = {} / {}".format(np.mean(f1s), np.std(f1s)))
    print(precisions)
    print(recalls)
    print(f1s)


if __name__ == '__main__':
    data_dir = './data/all'
    main(data_dir, dataset='IU', locations=['3T', '8T', '12T', '13T'], sbj_type='Healthy')
