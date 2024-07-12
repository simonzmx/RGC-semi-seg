import torch
import numpy as np
import tqdm
import time
from scipy import ndimage
from utils.external.guided_filter import GuidedFilter


def inference_whole_volume_segmentation(model, volume, data_cfg, num_classes, device):
    start_time = time.time()
    logits_sum = torch.zeros([num_classes] + list(volume.shape), dtype=torch.float32)
    probs_sum = torch.zeros([num_classes] + list(volume.shape), dtype=torch.float32)
    logits_tally = torch.zeros([num_classes] + list(volume.shape), dtype=torch.int16)

    # run inference as the original paper described
    z_size = int(data_cfg['window_size'][0])
    y_size = int(data_cfg['window_size'][1])
    x_size = int(data_cfg['window_size'][2])
    z_stride = int(data_cfg['window_stride'][0])
    y_stride = int(data_cfg['window_stride'][1])
    x_stride = int(data_cfg['window_stride'][2])
    z_steps = np.ceil((volume.shape[0] - z_size) / z_stride).astype(int) + 1
    y_steps = np.ceil((volume.shape[1] - y_size) / y_stride).astype(int) + 1
    x_steps = np.ceil((volume.shape[2] - x_size) / x_stride).astype(int) + 1

    for z_idx in range(z_steps):
        if z_idx * z_stride + z_size <= volume.shape[0]:
            z_range = range(z_idx * z_stride, z_idx * z_stride + z_size)
        else:
            z_range = range(volume.shape[0] - z_size, volume.shape[0])

        for y_idx in range(y_steps):
            if y_idx * y_stride + y_size <= volume.shape[1]:
                y_range = range(y_idx * y_stride, y_idx * y_stride + y_size)
            else:
                y_range = range(volume.shape[1] - y_size, volume.shape[1])

            for x_idx in range(x_steps):
                if x_idx * x_stride + x_size <= volume.shape[2]:
                    x_range = range(x_idx * x_stride, x_idx * x_stride + x_size)
                else:
                    x_range = range(volume.shape[2] - x_size, volume.shape[2])

                # get the sub-cube from the whole cube with the defined sliding window size
                this_sub_volume = volume[z_range.start: z_range.stop, y_range.start: y_range.stop, x_range.start: x_range.stop]

                # test-time augmentation, flips T/F + 4 rotation degrees
                for flip_idx in range(2):
                    for rot_idx in range(4):
                        aug_sub_volume = this_sub_volume
                        # apply flipping
                        if flip_idx > 0:
                            aug_sub_volume = np.flip(aug_sub_volume, axis=1)
                        # apply rotation
                        aug_sub_volume = np.rot90(aug_sub_volume, k=rot_idx, axes=(1, 2))
                        # expand channel dimension
                        aug_sub_volume = np.expand_dims(aug_sub_volume, axis=0).astype(np.float32)
                        # to tenser and expand batch dimension
                        aug_sub_volume = torch.unsqueeze(torch.tensor(aug_sub_volume), dim=0).to(device)
                        # model prediction
                        logits = model(aug_sub_volume)[0]  # dimension: class, z, y, x
                        # tensor move to cpu
                        logits = logits.detach().cpu()
                        # revert rotation and flipping
                        logits = torch.rot90(logits, k=-rot_idx, dims=[2, 3])
                        if flip_idx > 0:
                            logits = torch.flip(logits, dims=[2])
                        # prediction logits to probabilities
                        probs = torch.nn.functional.softmax(logits, dim=0)

                        logits_sum[:, z_range.start: z_range.stop, y_range.start: y_range.stop, x_range.start: x_range.stop] += logits
                        probs_sum[:, z_range.start: z_range.stop, y_range.start: y_range.stop, x_range.start: x_range.stop] += probs
                        logits_tally[:, z_range.start: z_range.stop, y_range.start: y_range.stop, x_range.start: x_range.stop] += 1

                        del logits, probs, aug_sub_volume

    logits_tally[logits_tally == 0] = 1
    logits_avg = logits_sum / logits_tally
    probs_avg = probs_sum / logits_tally
    print("Inference on the whole volume took {:.4f} seconds.".format(time.time() - start_time))
    return logits_avg, probs_avg


def inference_whole_volume_classification(model, volume, data_cfg, num_classes, device, batch_size=64):
    # run inference with sliding window
    patch_size = np.array(data_cfg['patch_size']).astype(np.int32)
    patch_half = np.floor(patch_size / 2).astype(np.int32)
    volume_pad = np.pad(volume, ((0, 0),
                                 (patch_half[1], patch_half[1]),
                                 (patch_half[2], patch_half[2])))
    logits_sum = torch.zeros([num_classes, volume.shape[0] - 2 * patch_half[0]] + list(volume.shape[1: 3]),
                             dtype=torch.float32)
    logits_tally = torch.zeros([num_classes, volume.shape[0] - 2 * patch_half[0]] + list(volume.shape[1: 3]),
                               dtype=torch.int16)

    batch_samples = np.floor(batch_size / 8).astype(np.int32)  # number of sub-volumes in a batch, division due to 8 test-time augmentations
    for z_idx in tqdm.tqdm(range(volume.shape[0] - 2 * patch_half[0])):
        for y_idx in range(volume.shape[1]):
            for batch_idx in range(np.ceil(volume.shape[2] / batch_samples).astype(np.int32)):
                x_range = np.arange(batch_idx * batch_samples,
                                    min(volume.shape[2], (batch_idx + 1) * batch_samples))
                batch_volume = []
                for x_idx in x_range:
                    # get the patch from the whole cube with the defined sliding window size
                    this_sub_volume = volume_pad[z_idx: z_idx + patch_size[0],
                                                 y_idx: y_idx + patch_size[1],
                                                 x_idx: x_idx + patch_size[2]]

                    # test-time augmentation, flips T/F + 4 rotation degrees
                    for flip_idx in range(2):
                        for rot_idx in range(4):
                            aug_sub_volume = this_sub_volume
                            # apply flipping
                            if flip_idx > 0:
                                aug_sub_volume = np.flip(aug_sub_volume, axis=1)
                            # apply rotation
                            aug_sub_volume = np.rot90(aug_sub_volume, k=rot_idx, axes=(1, 2))
                            # expand channel dimension
                            aug_sub_volume = np.expand_dims(aug_sub_volume, axis=0).astype(np.float32)
                            # append to the batch
                            batch_volume.append(aug_sub_volume)

                # model prediction
                this_sub_volume = torch.tensor(np.array(batch_volume)).to(device)
                preds = model(this_sub_volume)  # dimension: batch_size, class
                preds = preds.detach().cpu()

                # put predictions into the cube
                for x_idx in x_range:
                    logits_sum[:, z_idx, y_idx, x_idx] += torch.sum(preds[(x_idx - x_range[0]) * 8:
                                                                          (x_idx - x_range[0] + 1) * 8], dim=0)
                    logits_tally[:, z_idx, y_idx, x_idx] += 8

    logits_avg = logits_sum / logits_tally
    return logits_avg


def find_local_maximas(preds_md, local_maxima_filter):
    # local maxima filtering
    preds_lmax = ndimage.maximum_filter(preds_md, size=local_maxima_filter)

    # mask non-maxima voxels
    local_maxima_mask = (preds_md == preds_lmax)
    preds_lmax = preds_lmax * local_maxima_mask

    # get local maxima coordinates and values
    local_maxima_coords = np.argwhere(preds_lmax)
    local_maxima_values = preds_lmax[local_maxima_coords[:, 0], local_maxima_coords[:, 1], local_maxima_coords[:, 2]]

    # sort local maxima by value in descending order
    sorted_indices = np.argsort(local_maxima_values)[::-1]
    sorted_coords = local_maxima_coords[sorted_indices]

    # initialize an empty array for the output
    preds_lmax_unique = np.zeros_like(preds_lmax)

    # iterate over local maxima in descending order
    for coord in sorted_coords:
        # check if this local maxima has already been suppressed
        if preds_lmax[coord[0], coord[1], coord[2]] == 0:
            continue

        # add this local maxima to the output array
        preds_lmax_unique[coord[0], coord[1], coord[2]] = preds_lmax[coord[0], coord[1], coord[2]]

        # suppress all local maxima within the neighborhood of this local maxima
        preds_lmax[
            max(0, coord[0] - local_maxima_filter[0] // 2):min(preds_lmax.shape[0], coord[0] + local_maxima_filter[0] // 2 + 1),
            max(0, coord[1] - local_maxima_filter[1] // 2):min(preds_lmax.shape[1], coord[1] + local_maxima_filter[1] // 2 + 1),
            max(0, coord[2] - local_maxima_filter[2] // 2):min(preds_lmax.shape[2], coord[2] + local_maxima_filter[2] // 2 + 1)
        ] = 0

    return preds_lmax_unique


# def find_local_maximas_old(preds_md, local_maxima_filter):
#     # local maxima filtering
#     preds_lmax = ndimage.maximum_filter(preds_md, size=local_maxima_filter)
#
#     # mask non-maxima voxels and remove duplicates
#     for z_index in range(preds_lmax.shape[0]):
#         for y_index in range(preds_lmax.shape[1]):
#             for x_index in range(preds_lmax.shape[2]):
#                 if preds_lmax[z_index, y_index, x_index] == preds_md[z_index, y_index, x_index]:
#                     # mask surrounding voxels to prevent duplicated maximas
#                     preds_lmax[
#                         int(max(0, z_index - local_maxima_filter[0] // 2)):
#                         int(min(preds_md.shape[0], z_index + local_maxima_filter[0] // 2 + 1)),
#                         int(max(0, y_index - local_maxima_filter[1] // 2)):
#                         int(min(preds_md.shape[1], y_index + local_maxima_filter[1] // 2 + 1)),
#                         int(max(0, x_index - local_maxima_filter[2] // 2)):
#                         int(min(preds_md.shape[2], x_index + local_maxima_filter[2] // 2 + 1))
#                     ] = 0
#                     # keep only this voxel's value
#                     preds_lmax[z_index, y_index, x_index] = preds_md[z_index, y_index, x_index]
#                 else:
#                     # not local maxima
#                     preds_lmax[z_index, y_index, x_index] = 0
#
#     return preds_lmax


def segment_ganglion_cell(preds_map, elongated_gaussian_filter):
    gc_map = np.zeros(preds_map.shape)

    # self-guided filtering
    for z_index in range(preds_map.shape[0]):
        temp_img = preds_map[z_index, :, :]
        guided_filter = GuidedFilter(temp_img, 5, 0.01)
        guided_filter.filter(temp_img)
        gc_map[z_index, :, :] = temp_img

    # elongated Gaussian filter
    gc_map = ndimage.gaussian_filter(gc_map, sigma=elongated_gaussian_filter)

    # invert values
    gc_map = 1 - gc_map

    # TODO: paper page 4 (645)

    return gc_map


# if __name__ == '__main__':
#     a = np.random.rand(10, 10, 10)
#     b = [3, 3, 3]
#     md = ndimage.median_filter(a, b)
#     old_lm = find_local_maximas_old(md, b)
#     new_lm = find_local_maximas(md, b)
#     print(old_lm[old_lm != new_lm])
#     print(new_lm[old_lm != new_lm])
