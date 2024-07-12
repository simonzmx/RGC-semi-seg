import random
import numpy as np
import torch
from typing import Union, List, Tuple


def random_flip(input_array: np.array,
                label_array: np.array = None,
                labeled_mask: np.array = None,
                axis: Union[int, List[int]] = 0
                ) -> Tuple[np.array, np.array, np.array]:
    """ Randomly flip array and label along given axis/axes each with 0.5 probability."""
    flipped_array = input_array
    flipped_label = label_array
    flipped_mask = labeled_mask
    if type(axis) == int:
        axis = [axis]
    for this_axis in axis:
        if random.random() < 0.5:
            flipped_array = np.flip(flipped_array, this_axis)
            if flipped_label is not None:
                flipped_label = np.flip(flipped_label, this_axis)
            if flipped_mask is not None:
                flipped_mask = np.flip(flipped_mask, this_axis)
    return flipped_array, flipped_label, flipped_mask


def random_rotate3d_xy(input_array: np.array,
                       label_array: np.array = None,
                       labeled_mask: np.array = None,
                       ) -> Tuple[np.array, np.array, np.array]:
    """ Randomly rotate a 3d volume and label on the xy plane by a choice of (0, 90, 180, 270) degrees."""
    k = random.randrange(4)
    rotated_array = np.rot90(input_array, k=k, axes=(1, 2))
    if label_array is not None:
        rotated_label = np.rot90(label_array, k=k, axes=(1, 2))
    else:
        rotated_label = None
    if labeled_mask is not None:
        rotated_mask = np.rot90(labeled_mask, k=k, axes=(1, 2))
    else:
        rotated_mask = None
    return rotated_array, rotated_label, rotated_mask


def random_crop(input_array: np.array,
                crop_size: List[int],
                label_array: np.array = None,
                labeled_mask: np.array = None,
                ) -> Tuple[np.array, np.array, np.array]:
    # uniform random sampling
    z_rand = random.randrange(input_array.shape[0] - crop_size[0] + 1)
    y_rand = random.randrange(input_array.shape[1] - crop_size[1] + 1)
    x_rand = random.randrange(input_array.shape[2] - crop_size[2] + 1)

    # pre-difined windows (with strides)
    # z_stride = int(crop_size[0] / 2)
    # y_stride = int(crop_size[1] / 2)
    # x_stride = int(crop_size[2] / 2)
    # z_rand = random.randrange(np.ceil(input_array.shape[0] / z_stride)) * z_stride
    # y_rand = random.randrange(np.ceil(input_array.shape[1] / y_stride)) * y_stride
    # x_rand = random.randrange(np.ceil(input_array.shape[2] / x_stride)) * x_stride
    # z_rand = input_array.shape[0] - crop_size[0] if z_rand + crop_size[0] > input_array.shape[0] else z_rand
    # y_rand = input_array.shape[1] - crop_size[1] if y_rand + crop_size[1] > input_array.shape[1] else y_rand
    # x_rand = input_array.shape[2] - crop_size[2] if x_rand + crop_size[2] > input_array.shape[2] else x_rand

    # distance weights (or square of distance)
    # z_range = np.arange(0, input_array.shape[0] - crop_size[0] + 1)
    # z_weights = np.square(np.mean(z_range) - z_range) + 1
    # z_rand = random.choices(z_range, weights=z_weights, k=1)[0]
    # y_range = np.arange(0, input_array.shape[1] - crop_size[1] + 1)
    # y_weights = np.square(np.mean(y_range) - y_range) + 1
    # y_rand = random.choices(y_range, weights=y_weights, k=1)[0]
    # x_range = np.arange(0, input_array.shape[2] - crop_size[2] + 1)
    # x_weights = np.square(np.mean(x_range) - x_range) + 1
    # x_rand = random.choices(x_range, weights=x_weights, k=1)[0]

    # distance weights with cap
    # z_range = np.arange(0, input_array.shape[0] - crop_size[0] + 1)
    # z_weights = np.maximum(np.abs(np.mean(z_range) - z_range) - max((np.mean(z_range) - int(crop_size[0] / 2)), 0), 0 + 1
    # z_rand = random.choices(z_range, weights=z_weights, k=1)[0]
    # y_range = np.arange(0, input_array.shape[1] - crop_size[1] + 1)
    # y_weights = np.maximum(np.abs(np.mean(y_range) - y_range) - max((np.mean(y_range) - int(crop_size[1] / 2)), 0), 0) + 1
    # y_rand = random.choices(y_range, weights=y_weights, k=1)[0]
    # x_range = np.arange(0, input_array.shape[2] - crop_size[2] + 1)
    # x_weights = np.maximum(np.abs(np.mean(x_range) - x_range) - max((np.mean(x_range) - int(crop_size[2] / 2)), 0), 0) + 1
    # x_rand = random.choices(x_range, weights=x_weights, k=1)[0]

    # gaussian weights
    # z_range = np.arange(0, input_array.shape[0] - crop_size[0] + 1)
    # z_weights = norm.pdf(z_range, loc=np.mean(z_range), scale=np.std(z_range))
    # z_weights = 2 * np.max(z_weights) - z_weights
    # z_rand = random.choices(z_range, weights=z_weights, k=1)[0]
    # y_range = np.arange(0, input_array.shape[1] - crop_size[1] + 1)
    # y_weights = norm.pdf(y_range, loc=np.mean(y_range), scale=np.std(y_range))
    # y_weights = 2 * np.max(y_weights) - y_weights
    # y_rand = random.choices(y_range, weights=y_weights, k=1)[0]
    # x_range = np.arange(0, input_array.shape[2] - crop_size[2] + 1)
    # x_weights = norm.pdf(x_range, loc=np.mean(x_range), scale=np.std(x_range))
    # x_weights = 2 * np.max(x_weights) - x_weights
    # x_rand = random.choices(x_range, weights=x_weights, k=1)[0]

    # with open('D:\OSU\optometry\GC-Segmentation\src/tests\cropping_samples_mod_square_copy.txt', 'a') as f:
    #     f.write("{} {} {}\n".format(z_rand, y_rand, x_rand))

    volume_crop = input_array[z_rand: z_rand + crop_size[0], y_rand: y_rand + crop_size[1], x_rand: x_rand + crop_size[2]]

    if label_array is not None:
        label_crop = label_array[z_rand: z_rand + crop_size[0], y_rand: y_rand + crop_size[1], x_rand: x_rand + crop_size[2]]
    else:
        label_crop = None

    if labeled_mask is not None:
        mask_crop = labeled_mask[z_rand: z_rand + crop_size[0], y_rand: y_rand + crop_size[1], x_rand: x_rand + crop_size[2]]
    else:
        mask_crop = None
    return volume_crop, label_crop, mask_crop


def pad_volume(input_array: np.array,
               crop_size: List[int],
               label_array: np.array = None,
               labeled_mask: np.array = None,
               ) -> Tuple[np.array, np.array, np.array, np.array]:
    if np.all(np.array(input_array.shape) >= np.array(crop_size)):
        return input_array, input_array.shape, label_array, labeled_mask
    else:
        pad_size = [[max(np.floor((crop_size[index] - input_array.shape[index]) / 2).astype(np.int32), 0),
                     max(np.ceil((crop_size[index] - input_array.shape[index]) / 2).astype(np.int32), 0)]
                    for index in range(len(crop_size))]
        input_pad = np.pad(input_array, ((pad_size[0][0], pad_size[0][1]),
                                         (pad_size[1][0], pad_size[1][1]),
                                         (pad_size[2][0], pad_size[2][1])))
        if label_array is not None:
            label_pad = np.pad(label_array, ((pad_size[0][0], pad_size[0][1]),
                                             (pad_size[1][0], pad_size[1][1]),
                                             (pad_size[2][0], pad_size[2][1])))
        else:
            label_pad = None

        if labeled_mask is not None:
            mask_pad = np.pad(labeled_mask, ((pad_size[0][0], pad_size[0][1]),
                                             (pad_size[1][0], pad_size[1][1]),
                                             (pad_size[2][0], pad_size[2][1])))
        else:
            mask_pad = None
        return input_pad, input_array.shape, label_pad, mask_pad


def unpad_volume(input_array: Union[np.array, torch.Tensor],
                 original_size: List[int],
                 ) -> Union[np.array, torch.Tensor]:
    pad_size = [max(np.floor((input_array.shape[-3 + index] - original_size[index]) / 2).astype(np.int32), 0)
                for index in range(len(original_size))]
    input_unpad = input_array[...,
                              pad_size[0]: pad_size[0] + original_size[0],
                              pad_size[1]: pad_size[1] + original_size[1],
                              pad_size[2]: pad_size[2] + original_size[2]]
    return input_unpad


def get_sphere_mask(radius: int,
                    voxel_resolution: List[float],
                    ) -> np.array:
    sizes = [np.around(radius / res) * 2 + 1 for res in voxel_resolution]
    centers = [np.around(radius / res) for res in voxel_resolution]
    z, y, x = np.mgrid[0:sizes[0]:1, 0:sizes[1]:1, 0:sizes[2]:1]
    sphere = np.sqrt(((z - centers[0]) * voxel_resolution[0]) ** 2
                     + ((y - centers[1]) * voxel_resolution[1]) ** 2
                     + ((x - centers[2]) * voxel_resolution[2]) ** 2)
    mask = np.zeros(sphere.shape)
    mask[sphere <= radius] = 1
    return mask


def insert_cell_sphere(label: np.array,
                       cord: Union[List[int], np.array],
                       sphere_mask: np.array
                       ) -> np.array:
    radius = [size // 2 for size in sphere_mask.shape]
    label_start_z = max(0, cord[0] - radius[0])
    label_end_z = min(label.shape[0], cord[0] + radius[0] + 1)
    label_start_y = max(0, cord[1] - radius[1])
    label_end_y = min(label.shape[1], cord[1] + radius[1] + 1)
    label_start_x = max(0, cord[2] - radius[2])
    label_end_x = min(label.shape[2], cord[2] + radius[2] + 1)
    if label_start_z > label_end_z or label_start_y > label_end_y or label_start_x > label_end_x:
        return label

    mask_start_z = label_start_z - (cord[0] - radius[0])
    mask_end_z = sphere_mask.shape[0] - (cord[0] + radius[0] + 1 - label_end_z)
    mask_start_y = label_start_y - (cord[1] - radius[1])
    mask_end_y = sphere_mask.shape[1] - (cord[1] + radius[1] + 1 - label_end_y)
    mask_start_x = label_start_x - (cord[2] - radius[2])
    mask_end_x = sphere_mask.shape[2] - (cord[2] + radius[2] + 1 - label_end_x)

    label[label_start_z: label_end_z, label_start_y: label_end_y, label_start_x: label_end_x] = \
        np.maximum(label[label_start_z: label_end_z, label_start_y: label_end_y, label_start_x: label_end_x],
                   sphere_mask[mask_start_z: mask_end_z, mask_start_y: mask_end_y, mask_start_x: mask_end_x])
    return label


def normalize_volume(vol: np.array,
                     mean: float = None,
                     std: float = None
                     ) -> np.array:
    vol = np.copy(vol)
    if mean is None:
        mean = np.mean(vol)
    if std is None:
        std = np.std(vol)
    if std != 0:
        vol = (vol - mean) / std
    return vol


def _blend(img1: torch.Tensor, img2: torch.Tensor, ratio: float) -> torch.Tensor:
    ratio = float(ratio)
    return ((1.0 - ratio) * img1 + ratio * img2).clamp(0, 255).to(img1.dtype)


def adjust_brightness(img: torch.Tensor, brightness_factor: float) -> torch.Tensor:
    if brightness_factor < 0:
        raise ValueError(f"brightness_factor ({brightness_factor}) is not non-negative.")
    if torch.rand(1) < 0.5:
        img2 = torch.zeros_like(img)
    else:
        img2 = torch.ones_like(img) * 255
    return _blend(img, img2, brightness_factor)


def adjust_contrast(img: torch.Tensor, contrast_factor: float) -> torch.Tensor:
    if contrast_factor < 0:
        raise ValueError(f"contrast_factor ({contrast_factor}) is not non-negative.")
    dtype = img.dtype if torch.is_floating_point(img) else torch.float32
    mean = torch.mean(img.to(dtype), dim=(-3, -2, -1), keepdim=True)
    return _blend(img, mean, contrast_factor)


class ColorJitter3D(torch.nn.Module):

    def __init__(self, brightness: float = 0, contrast: float = 0):
        super(ColorJitter3D, self).__init__()
        self.brightness = brightness
        self.contrast = contrast

    def forward(self, img):
        fn_idx = torch.randperm(2)
        brightness_factor = float(torch.empty(1).uniform_(0, self.brightness))
        contrast_factor = float(torch.empty(1).uniform_(0, self.contrast))

        for fn_id in fn_idx:
            if fn_id == 0 and brightness_factor is not None:
                img = adjust_brightness(img, brightness_factor)
            elif fn_id == 1 and contrast_factor is not None:
                img = adjust_contrast(img, contrast_factor)

        return img


if __name__ == '__main__':
    # m = np.arange(24).reshape((2, 3, 4))
    # n = -m
    # # rm, rn = random_flip(n, m, [0, 1, 2])
    # rm, rn = random_rotate3d_xy(n, label_array=m)
    # print('rm:')
    # print(rm)
    # print('rn:')
    # print(rn)

    # m = np.random.randint(5, size=(9, 2, 5))
    # n = -m
    # m_pad, ori_size, n_pad, _ = pad_volume(m, [5, 7, 7], label_array=n)
    # m_pad_4d = np.expand_dims(m_pad, axis=[0, 1])
    # m_ori = unpad_volume(m_pad_4d, ori_size)
    # print(m_pad)
    # print(m_pad.shape)
    # print(n_pad)
    # print(n_pad.shape)
    # print(m_ori)
    # print(m_ori.shape)

    radius = 2
    # voxel_res = [0.685, 1.5, 1.5]
    voxel_res = [0.94, 0.97, 0.97]
    print(get_sphere_mask(radius, voxel_res))
