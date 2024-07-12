import numpy as np


def build_gt2pred_ordered_mapping(gt_coords, preds_lmax, voxel_res, dis_threshold):
    # O(n*m), n: number of gt_coords, m: number of preds_lmax
    gt2pred_ordered_map = {}
    pred_coords = np.array(np.nonzero(preds_lmax)).transpose()
    for gt_idx, gt_coord in enumerate(gt_coords):
        available_preds = []
        for pred_idx, pred_coord in enumerate(pred_coords):
            # calculate ground-truth to prediction distance
            distance = np.sqrt((voxel_res[0] * (gt_coord[0] - pred_coord[0])) ** 2 +
                               (voxel_res[1] * (gt_coord[1] - pred_coord[1])) ** 2 +
                               (voxel_res[2] * (gt_coord[2] - pred_coord[2])) ** 2)
            if distance < dis_threshold:
                available_preds.append((pred_idx,  # index
                                        preds_lmax[pred_coord[0], pred_coord[1], pred_coord[2]],  # probability
                                        distance))  # distance
        preds_sorted = sorted(available_preds, key=lambda x: x[2])  # sorted by distance
        gt2pred_ordered_map[gt_idx] = preds_sorted
    return gt2pred_ordered_map


def measure_performance(preds_lmax, gt_coords, volume_shape, voxel_res, dis_threshold, border_size=10, mode='val'):
    pred_coords = np.array(np.nonzero(preds_lmax)).transpose()
    print(f'Predicted GCs number={pred_coords.shape[0]}, ground-truth GCs number=({gt_coords.shape[0]}).')
    if mode == 'val' and pred_coords.shape[0] > gt_coords.shape[0] * 100:
        print(f'Skip evaluation due to too many predicted GCs compared to ground-truth.')
        best_index = 0
        precisions, recalls, f1s, thresholds = [0], [0], [0], [0]
    else:
        gt2pred_ordered_map = build_gt2pred_ordered_mapping(gt_coords, preds_lmax, voxel_res, dis_threshold)
        precisions, recalls, f1s = [], [], []
        thresholds = np.arange(0, 1, .001)[::-1]
        for threshold in thresholds:
            gt_matched = np.zeros(gt_coords.shape[0], dtype=np.int32)
            pred_matched = np.zeros(pred_coords.shape[0], dtype=np.int32)
            for pred_idx, pred_coord in enumerate(pred_coords):
                # discard predictions with probability lower than the threshold
                if preds_lmax[pred_coord[0], pred_coord[1], pred_coord[2]] < threshold:
                    pred_matched[pred_idx] = -1

            for gt_idx, gt_coord in enumerate(gt_coords):
                if gt_matched[gt_idx] == 0:
                    for matched_pred in gt2pred_ordered_map[gt_idx]:
                        if pred_matched[matched_pred[0]] == 0:  # this prediction is available
                            gt_matched[gt_idx] = matched_pred[0] + 1  # keep 0 indicating non-matched, thus +1
                            pred_matched[matched_pred[0]] = gt_idx + 1  # keep 0 indicating non-matched, thus +1
                            break

            # disregard border voxels
            for pred_idx, pred_coord in enumerate(pred_coords):
                if not (border_size <= pred_coord[0] < volume_shape[0] - border_size and
                        border_size <= pred_coord[1] < volume_shape[1] - border_size and
                        border_size <= pred_coord[2] < volume_shape[2] - border_size):
                    if pred_matched[pred_idx] > 0:
                        gt_matched[pred_matched[pred_idx] - 1] = -1
                    pred_matched[pred_idx] = -1
            for gt_idx, gt_coord in enumerate(gt_coords):
                if not (border_size <= gt_coord[0] < volume_shape[0] - border_size and
                        border_size <= gt_coord[1] < volume_shape[1] - border_size and
                        border_size <= gt_coord[2] < volume_shape[2] - border_size):
                    if gt_matched[gt_idx] > 0:
                        pred_matched[gt_matched[gt_idx] - 1] = -1
                    gt_matched[gt_idx] = -1

            # calculate precision and recall
            true_pos = np.count_nonzero(gt_matched > 0)
            false_pos = np.count_nonzero(pred_matched == 0)
            false_neg = np.count_nonzero(gt_matched == 0)
            pre = true_pos / (true_pos + false_pos + 1e-6)
            rec = true_pos / (true_pos + false_neg + 1e-6)
            f1 = 2 * pre * rec / (pre + rec + 1e-6)
            precisions.append(pre)
            recalls.append(rec)
            f1s.append(f1)

        best_index = f1s.index(max(f1s))
        avg_pre = calculate_average_precisions(precisions, recalls)
        print(f'Average precision={avg_pre:.4f} '
              f'Precision={precisions[best_index]:.4f} recall={recalls[best_index]:.4f} '
              f'f1-score={f1s[best_index]:.4f} threshold={thresholds[best_index]:.4f}')

    return precisions, recalls, f1s, best_index, thresholds[best_index]


def calculate_average_precisions(precisions, recalls):
    assert len(precisions) == len(recalls)

    # Sort by recalls
    sorted_indices = np.argsort(recalls)
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]

    # Interpolate precision values
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # Add data point for recall=0
    recalls = [0] + recalls
    precisions = [max(precisions)] + precisions

    # Add data point for recall=1
    recalls.append(1)
    precisions.append(0)

    avg_pres = 0
    for idx in range(1, len(recalls)):
        avg_pres += precisions[idx] * (recalls[idx] - recalls[idx - 1])

    return avg_pres


def measure_performance_pred_thresholded(pred_coords, gt_coords, volume_shape, voxel_res, dis_threshold, border_size=10):
    gt_matched = np.zeros(gt_coords.shape[0], dtype=np.int32)
    pred_matched = np.zeros(pred_coords.shape[0], dtype=np.int32)

    # match each ground-truth coordinates to the nearest predicted coordinates
    for gt_idx, gt_coord in enumerate(gt_coords):
        if gt_matched[gt_idx] >= 0:
            min_dis = np.inf
            min_idx = -1
            for pred_idx, pred_coord in enumerate(pred_coords):
                if pred_matched[pred_idx] == 0:
                    # this prediction coordinate is available
                    distance = np.sqrt((voxel_res[0] * (gt_coord[0] - pred_coord[0])) ** 2 +
                                       (voxel_res[1] * (gt_coord[1] - pred_coord[1])) ** 2 +
                                       (voxel_res[2] * (gt_coord[2] - pred_coord[2])) ** 2)
                    if distance < dis_threshold and distance < min_dis:
                        min_dis = distance
                        min_idx = pred_idx
            if min_idx > -1:
                gt_matched[gt_idx] = min_idx + 1  # keep 0 indicating non-matched, thus +1
                pred_matched[min_idx] = gt_idx + 1  # keep 0 indicating non-matched, thus +1

    # disregard border voxels
    for pred_idx, pred_coord in enumerate(pred_coords):
        if not (border_size <= pred_coord[0] < volume_shape[0] - border_size and
                border_size <= pred_coord[1] < volume_shape[1] - border_size and
                border_size <= pred_coord[2] < volume_shape[2] - border_size):
            if pred_matched[pred_idx] > 0:
                gt_matched[pred_matched[pred_idx] - 1] = -1
            pred_matched[pred_idx] = -1
    for gt_idx, gt_coord in enumerate(gt_coords):
        if not (border_size <= gt_coord[0] < volume_shape[0] - border_size and
                border_size <= gt_coord[1] < volume_shape[1] - border_size and
                border_size <= gt_coord[2] < volume_shape[2] - border_size):
            if gt_matched[gt_idx] > 0:
                pred_matched[gt_matched[gt_idx] - 1] = -1
            gt_matched[gt_idx] = -1

    # calculate precision and recall
    true_pos = np.count_nonzero(gt_matched > 0)
    false_pos = np.count_nonzero(pred_matched == 0)
    false_neg = np.count_nonzero(gt_matched == 0)
    precision = true_pos / (true_pos + false_pos + 1e-6)
    recall = true_pos / (true_pos + false_neg + 1e-6)
    f1 = 2 * precision * recall / (precision + recall + 1e-6)
    print(f'Precision={precision:.4f} recall={recall:.4f} f1-score={f1:.4f}')
    return precision, recall, f1, gt_matched, pred_matched
