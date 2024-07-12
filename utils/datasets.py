import os
import glob
import random
import json
import torch
import numpy as np
import scipy.io as scio
import skimage.io as skio
from collections import OrderedDict
from torch.utils.data import Dataset
from utils.transforms import random_flip, random_rotate3d_xy, random_crop, get_sphere_mask, insert_cell_sphere,\
    normalize_volume, pad_volume, ColorJitter3D
from torchvision import transforms
# from utils.visualizations import visualize_volume, visualize_volume_mid_xy_yz_planes


def get_img_paths_by_mode(data_cfg: dict, mode: str):
    img_paths = glob.glob(os.path.join(data_cfg['root_dir'], 'images', '*.tif'))
    train_subjects = data_cfg['all_subjects'].copy()
    train_subjects = [x for x in train_subjects if x not in data_cfg['test_subjects']]
    train_subjects = [x for x in train_subjects if x not in data_cfg['val_subjects']]

    selected_paths = OrderedDict()
    for img_path in img_paths:
        img_filename = img_path.split(os.sep)[-1][:-4]
        this_dataset, this_subject, this_location = img_filename.split('_')[0: 3]
        if this_dataset not in data_cfg["dataset"]:
            continue

        if mode == 'train':
            if (this_subject in train_subjects) \
                    or (this_subject in data_cfg['val_subjects'] and this_location != '12T' and this_location != '13T'):
                selected_paths[img_path] = data_cfg['location_probs'][this_location] \
                    if 'location_probs' in data_cfg and this_location in data_cfg['location_probs'] else 1
        elif mode == 'val':
            if this_subject in data_cfg['val_subjects'] and (this_location == '12T' or this_location == '13T'):
                selected_paths[img_path] = data_cfg['location_probs'][this_location] \
                    if 'location_probs' in data_cfg and this_location in data_cfg['location_probs'] else 1
        elif mode == 'test':
            if this_subject in data_cfg['test_subjects']:
                selected_paths[img_path] = data_cfg['location_probs'][this_location] \
                    if 'location_probs' in data_cfg and this_location in data_cfg['location_probs'] else 1
        else:
            raise ValueError('Mode can only be \'train\' or \'val\' or \'test\'!')
    return selected_paths


def select_labeled_img_paths(img_paths, data_cfg):
    subject_paths = {}
    for img_path, img_prob in img_paths.items():
        img_filename = img_path.split(os.sep)[-1][:-4]
        this_dataset, this_subject, this_location = img_filename.split('_')[0: 3]
        if this_subject not in subject_paths:
            subject_paths[this_subject] = {img_path: img_prob}
        else:
            subject_paths[this_subject][img_path] = img_prob

    if 'train_labeled_subjects' in data_cfg:
        selected_subjects = data_cfg['train_labeled_subjects']
    else:
        selected_subjects = random.sample(
            list(subject_paths.keys()),
            k=np.ceil(len(subject_paths.keys()) * data_cfg['label_usage_ratio']).astype(np.int32)
        )
    print('Selected subjects used for supervised training:')
    print(selected_subjects)

    selected_paths = {}
    for this_subject in selected_subjects:
        if this_subject in subject_paths:
            selected_paths.update(subject_paths[this_subject])
    return selected_paths


class GCSegmentationDataset(Dataset):
    def __init__(self, data_cfg, mode='train', dataset_type='semi-supervised'):
        self.mode = mode
        self.dataset_type = dataset_type
        if dataset_type != 'supervised' and dataset_type != 'semi-supervised':
            raise ValueError('\'dataset_type\' must be \'supervised\' or \'semi-supervised\'.')
        # with open('D:\OSU\optometry\GC-Segmentation\src/tests\cropping_samples_mod_square_copy.txt', 'a') as f:
        #     f.write("{}\n".format(this_img_path))

        self.root_path = data_cfg['root_dir']
        self.img_paths = get_img_paths_by_mode(data_cfg, 'train')
        self.img_paths_labeled = select_labeled_img_paths(self.img_paths, data_cfg)
        self.epoch_samples = data_cfg['epoch_samples_train']

        self.crop_size = data_cfg['crop_size']
        self.normalizations = data_cfg['normalizations']
        self.cell_radius = data_cfg['cell_radius']
        self.sphere_mask = get_sphere_mask(self.cell_radius, data_cfg['voxel_resolution'])

        self.strong_augmentation = ColorJitter3D(0.5, 0.5)

        with open(data_cfg['layer_seg_info'], 'r') as f:
            self.layer_info = json.loads(f.read())

    def __len__(self):
        return self.epoch_samples    # number of samples per epoch

    def __getitem__(self, idx):
        # Randomly select one image
        # obeying "Set the probability of selecting the 13T volumes to be five times higher than the 3T volumes."
        this_img_path = random.choices(list(self.img_paths_labeled.keys()), weights=list(self.img_paths_labeled.values()), k=1)[0]

        # load this volume
        volume_ori = skio.imread(this_img_path).astype(np.float64)   # volume axis: y, z, x
        volume_ori = np.transpose(volume_ori, axes=(1, 0, 2))  # volume axis: z, y, x

        # load label coordinates
        file_identifier = this_img_path.split(os.sep)[-1][:-4]
        label_file_path = os.path.join(self.root_path, 'labels', file_identifier + '_marker.mat')
        mat_content = scio.loadmat(label_file_path)
        gc_coords = mat_content['temp_marker'].astype(np.int32)    # coordinates axis: z, x, y
        gc_coords[:, [1, 2]] = gc_coords[:, [2, 1]]  # coordinates axis: z, y, x
        gc_coords -= 1    # original coordinates start from 1, make it start from 0

        # crop (z-axis) at the retinal layer region
        # TODO: Ganglion Cell Layer segmentation according to WeakGCSeg paper
        if file_identifier in self.layer_info:
            min_z = max(self.layer_info[file_identifier]['min_z'], 0)
            max_z = min(self.layer_info[file_identifier]['max_z'] + 1, volume_ori.shape[0])
        else:
            min_z = max(np.min(gc_coords[:, 0]), 0)
            max_z = min(np.max(gc_coords[:, 0]) + 1, volume_ori.shape[0])
        volume = volume_ori[min_z: max_z, :, :]
        gc_crop_coords = gc_coords - np.array([min_z, 0, 0])  # convert gc coordinates to the cropped volume

        # build label volume
        label = np.zeros(volume.shape).astype(np.uint8)
        for coord_idx in range(gc_crop_coords.shape[0]):
            label = insert_cell_sphere(label, gc_crop_coords[coord_idx, :], self.sphere_mask)

        # add gaussian noise
        volume += np.random.normal(0, 1.5, size=volume.shape)
        # determine mean and std
        this_mean = self.normalizations[0] if self.normalizations[0] is not None else np.mean(volume)
        this_std = self.normalizations[1] if self.normalizations[1] is not None else np.std(volume)
        # pad volume if needed (volume size smaller than window size)
        volume, ori_size, label, _ = pad_volume(volume, self.crop_size, label_array=label)
        # normalize input volume
        volume = normalize_volume(volume, mean=this_mean, std=this_std)
        # random crop
        volume, label, _ = random_crop(volume, self.crop_size, label_array=label)
        # random rotation
        volume, label, _ = random_rotate3d_xy(volume, label_array=label)
        # random flip
        volume, label, _ = random_flip(volume, label_array=label, axis=[0, 1, 2])

        volume = torch.from_numpy(np.expand_dims(volume, axis=0).astype(np.float32))
        label = torch.from_numpy(np.array([1 - label, label]).astype(np.float32))

        # test_v = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.int8)
        # # test_l = (label * 255).astype(np.int8)
        # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_v.tiff", test_v[0])
        # # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_l.tiff", test_l[1])

        if self.dataset_type == 'semi-supervised':
            # Randomly select one image
            # obeying "Set the probability of selecting the 13T volumes to be five times higher than the 3T volumes."
            this_img_path = random.choices(list(self.img_paths.keys()), weights=list(self.img_paths.values()), k=1)[0]
            file_identifier = this_img_path.split(os.sep)[-1][:-4]
            volume_ori = skio.imread(this_img_path).astype(np.float64)  # volume axis: y, z, x
            volume_ori = np.transpose(volume_ori, axes=(1, 0, 2))  # volume axis: z, y, x
            if file_identifier in self.layer_info:
                min_z = max(self.layer_info[file_identifier]['min_z'], 0)
                max_z = min(self.layer_info[file_identifier]['max_z'] + 1, volume_ori.shape[0])
            else:
                label_file_path = os.path.join(self.root_path, 'labels', this_img_path.split(os.sep)[-1][:-4] + '_marker.mat')
                mat_content = scio.loadmat(label_file_path)
                gc_coords = mat_content['temp_marker'].astype(np.int32)  # coordinates axis: z, x, y
                gc_coords[:, [1, 2]] = gc_coords[:, [2, 1]]  # coordinates axis: z, y, x
                gc_coords -= 1  # original coordinates start from 1, make it start from 0
                min_z = max(np.min(gc_coords[:, 0]), 0)
                max_z = min(np.max(gc_coords[:, 0]) + 1, volume_ori.shape[0])
            volume_uw = volume_ori[min_z: max_z, :, :]

            # add noise
            volume_uw += np.random.normal(0, 1.5, size=volume_uw.shape)
            # determine mean and std
            this_mean = self.normalizations[0] if self.normalizations[0] is not None else np.mean(volume_uw)
            this_std = self.normalizations[1] if self.normalizations[1] is not None else np.std(volume_uw)
            volume_uw, ori_size, _, _ = pad_volume(volume_uw, self.crop_size)
            volume_uw, _, _ = random_crop(volume_uw, self.crop_size)
            # weak augmentations: flip and rotate
            volume_uw, _, _ = random_rotate3d_xy(volume_uw)
            volume_uw, _, _ = random_flip(volume_uw, axis=[0, 1, 2])
            volume_uw = torch.from_numpy(np.expand_dims(volume_uw, axis=0).astype(np.float32))
            # strong augmentation: color jitter
            volume_us = self.strong_augmentation(volume_uw)
            # normalize
            volume_uw = transforms.Normalize(mean=this_mean, std=this_std)(volume_uw)
            volume_us = transforms.Normalize(mean=this_mean, std=this_std)(volume_us)
        else:
            volume_uw = torch.empty(0)
            volume_us = torch.empty(0)

        # import imageio
        # test_lv = volume.numpy().clip(0, 255).astype(np.int8)
        # # test_ll = label.astype(np.int8) * 255
        # # test_lm = labeled_mask.astype(np.int8) * 255
        # test_uw = volume_uw.numpy().clip(0, 255).astype(np.int8)
        # test_us = volume_us.numpy().clip(0, 255).astype(np.int8)
        # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_lv.tiff", test_lv[0])
        # # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_ll.tiff", test_ll[1])
        # # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_lm.tiff", test_lm)
        # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_uw.tiff", test_uw[0])
        # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_us.tiff", test_us[0])

        return volume, label, volume_uw, volume_us


class GCSegmentationSparseDataset(Dataset):
    def __init__(self, data_cfg, mode='train', dataset_type='semi-supervised'):
        self.mode = mode
        self.dataset_type = dataset_type
        if dataset_type != 'supervised' and dataset_type != 'semi-supervised':
            raise ValueError('\'dataset_type\' must be \'supervised\' or \'semi-supervised\'.')

        self.root_path = data_cfg['root_dir']
        self.img_paths = get_img_paths_by_mode(data_cfg, 'train')
        self.label_usage_ratio = data_cfg['label_usage_ratio']
        self.epoch_samples = data_cfg['epoch_samples_train']

        self.crop_size = data_cfg['crop_size']
        self.normalizations = data_cfg['normalizations']
        self.cell_radius = data_cfg['cell_radius']
        self.sphere_mask = get_sphere_mask(self.cell_radius, data_cfg['voxel_resolution'])

        self.strong_augmentation = ColorJitter3D(0.5, 0.5)

        with open(data_cfg['layer_seg_info'], 'r') as f:
            self.layer_info = json.loads(f.read())

    def __len__(self):
        return self.epoch_samples    # number of samples per epoch

    def __getitem__(self, idx):
        # Randomly select one image
        # obeying "Set the probability of selecting the 13T volumes to be five times higher than the 3T volumes."
        this_img_path = random.choices(list(self.img_paths.keys()), weights=list(self.img_paths.values()), k=1)[0]
        file_identifier = this_img_path.split(os.sep)[-1][:-4]

        # load this volume
        volume_ori = skio.imread(this_img_path).astype(np.float64)   # volume axis: y, z, x
        volume_ori = np.transpose(volume_ori, axes=(1, 0, 2))  # volume axis: z, y, x

        # load label coordinates
        label_file_path = os.path.join(self.root_path, 'labels_{:.2f}'.format(self.label_usage_ratio), file_identifier + '.npy')
        gc_coords = np.load(label_file_path).astype(np.int32)  # coordinates axis: z, y, x, class

        # crop (z-axis) at the retinal layer region
        # TODO: Ganglion Cell Layer segmentation according to WeakGCSeg paper
        if file_identifier in self.layer_info:
            min_z = max(self.layer_info[file_identifier]['min_z'], 0)
            max_z = min(self.layer_info[file_identifier]['max_z'] + 1, volume_ori.shape[0])
        else:
            min_z = max(np.min(gc_coords[:, 0]), 0)
            max_z = min(np.max(gc_coords[:, 0]) + 1, volume_ori.shape[0])
        volume = volume_ori[min_z: max_z, :, :]
        gc_crop_coords = gc_coords - np.array([min_z, 0, 0, 0])  # convert gc coordinates to the cropped volume

        # build label and mask volume
        label = np.zeros(volume.shape).astype(np.uint8)
        labeled_mask = np.zeros(volume.shape).astype(np.uint8) if self.label_usage_ratio < 1 \
            else np.ones(volume.shape).astype(np.uint8)
        for coord_idx in range(gc_crop_coords.shape[0]):
            if gc_crop_coords[coord_idx, 3] == 1:
                # insert ganglion cell sphere
                label = insert_cell_sphere(label, gc_crop_coords[coord_idx, 0: 3], self.sphere_mask)
            if self.label_usage_ratio < 1:
                # insert sampled pos/neg sphere into the mask
                labeled_mask = insert_cell_sphere(labeled_mask, gc_crop_coords[coord_idx, 0: 3], self.sphere_mask)

        # add gaussian noise
        volume += np.random.normal(0, 1.5, size=volume.shape)
        # determine mean and std
        this_mean = self.normalizations[0] if self.normalizations[0] is not None else np.mean(volume)
        this_std = self.normalizations[1] if self.normalizations[1] is not None else np.std(volume)
        # pad volume if needed (volume size smaller than window size)
        volume, ori_size, label, labeled_mask = pad_volume(volume, self.crop_size, label_array=label, labeled_mask=labeled_mask)
        # normalize input volume
        volume = normalize_volume(volume, mean=this_mean, std=this_std)

        error_counter = 0
        while True:
            # random crop
            volume_cropped, label_cropped, labeled_mask_cropped = \
                random_crop(volume, self.crop_size, label_array=label, labeled_mask=labeled_mask)
            # only use a valid cropping when cropped volume has available positive and negative samples
            if np.sum(label_cropped[labeled_mask_cropped == 1] == 1) > np.sum(self.sphere_mask) and \
                np.sum(label_cropped[labeled_mask_cropped == 1] == 1) > np.sum(self.sphere_mask):
                break
            error_counter += 1
            if error_counter > 1000:
                raise ValueError('File {} does not have enough labeled samples when label usage ratio={}.'.format(this_img_path, self.label_usage_ratio))

        # random rotation
        volume, label, labeled_mask = random_rotate3d_xy(volume_cropped, label_array=label_cropped,
                                                         labeled_mask=labeled_mask_cropped)
        # random flip
        volume, label, labeled_mask = random_flip(volume, label_array=label, labeled_mask=labeled_mask, axis=[0, 1, 2])

        volume = np.expand_dims(volume, axis=0).astype(np.float32)
        label = np.array([1 - label, label]).astype(np.float32)
        labeled_mask = labeled_mask.astype(np.float32)

        # temp_mask = np.tile(labeled_mask, [label.shape[0], 1, 1, 1])
        # label[temp_mask == 0] = 0
        # print(np.sum(label[temp_mask == 1]))

        if self.dataset_type == 'semi-supervised':
            # Randomly select one image
            # obeying "Set the probability of selecting the 13T volumes to be five times higher than the 3T volumes."
            this_img_path = random.choices(list(self.img_paths.keys()), weights=list(self.img_paths.values()), k=1)[0]
            file_identifier = this_img_path.split(os.sep)[-1][:-4]
            volume_ori = skio.imread(this_img_path).astype(np.float64)  # volume axis: y, z, x
            volume_ori = np.transpose(volume_ori, axes=(1, 0, 2))  # volume axis: z, y, x
            if file_identifier in self.layer_info:
                min_z = max(self.layer_info[file_identifier]['min_z'], 0)
                max_z = min(self.layer_info[file_identifier]['max_z'] + 1, volume_ori.shape[0])
            else:
                label_file_path = os.path.join(self.root_path, 'labels',
                                               this_img_path.split(os.sep)[-1][:-4] + '_marker.mat')
                mat_content = scio.loadmat(label_file_path)
                gc_coords = mat_content['temp_marker'].astype(np.int32)  # coordinates axis: z, x, y
                gc_coords[:, [1, 2]] = gc_coords[:, [2, 1]]  # coordinates axis: z, y, x
                gc_coords -= 1  # original coordinates start from 1, make it start from 0
                min_z = max(np.min(gc_coords[:, 0]), 0)
                max_z = min(np.max(gc_coords[:, 0]) + 1, volume_ori.shape[0])
            volume_uw = volume_ori[min_z: max_z, :, :]

            # add noise
            volume_uw += np.random.normal(0, 1.5, size=volume_uw.shape)
            # determine mean and std
            this_mean = self.normalizations[0] if self.normalizations[0] is not None else np.mean(volume_uw)
            this_std = self.normalizations[1] if self.normalizations[1] is not None else np.std(volume_uw)
            volume_uw, ori_size, _, _ = pad_volume(volume_uw, self.crop_size)
            volume_uw, _, _ = random_crop(volume_uw, self.crop_size)
            # weak augmentations: flip and rotate
            volume_uw, _, _ = random_rotate3d_xy(volume_uw)
            volume_uw, _, _ = random_flip(volume_uw, axis=[0, 1, 2])
            volume_uw = torch.from_numpy(np.expand_dims(volume_uw, axis=0).astype(np.float32))
            # strong augmentation: color jitter
            volume_us = self.strong_augmentation(volume_uw)
            # normalize
            volume_uw = transforms.Normalize(mean=this_mean, std=this_std)(volume_uw)
            volume_us = transforms.Normalize(mean=this_mean, std=this_std)(volume_us)
        else:
            volume_uw = torch.empty(0)
            volume_us = torch.empty(0)

        # import imageio
        # test_lv = ((volume - volume.min()) / (volume.max() - volume.min()) * 255).astype(np.int8)
        # test_ll = label.astype(np.int8) * 255
        # test_lm = labeled_mask.astype(np.int8) * 255
        # test_uv = ((volume_unlabeled - volume_unlabeled.min()) / (volume_unlabeled.max() - volume_unlabeled.min()) * 255).astype(np.int8)
        # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_lv.tiff", test_lv[0])
        # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_ll.tiff", test_ll[1])
        # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_lm.tiff", test_lm)
        # imageio.mimwrite(f"D:\OSU\optometry\GC-Segmentation\src/tests/samples/{idx:04d}_uv.tiff", test_uv[0])

        return volume, label, labeled_mask, volume_uw, volume_us
