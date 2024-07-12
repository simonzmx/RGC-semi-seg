import os
import json
from PIL import Image
import numpy as np
import scipy.io as scio
import skimage.io as skio
from utils.transforms import random_flip, random_rotate3d_xy, normalize_volume


def main(data_dir='./data/all/labels'):
    limits_dict = {}
    for filename in os.listdir(data_dir):
        if filename.endswith('.mat'):
            # load label coordinates
            label_file_path = os.path.join(data_dir, filename)
            file_prefix = '_'.join(filename.split('_')[:-1])
            mat_content = scio.loadmat(label_file_path)
            gc_coords = mat_content['temp_marker'].astype(np.int32)  # coordinates axis: z, x, y
            gc_coords[:, [1, 2]] = gc_coords[:, [2, 1]]  # coordinates axis: z, y, x
            gc_coords -= 1  # original coordinates start from 1, make it start from 0

            # crop (z-axis) at the retinal layer region
            # TODO: do this step as the original paper, not using the ground-truth
            min_z = int(np.min(gc_coords[:, 0]))
            max_z = int(np.max(gc_coords[:, 0]))
            limits_dict[file_prefix] = {'min_z': min_z, 'max_z': max_z}

    with open('./data/all/gc_layers.json', 'w') as f:
        f.write(json.dumps(limits_dict))


def modify_layer_region(info_file='./data/all/gc_layers_reduced.json', size=10):
    with open(info_file, 'r') as f:
        layer_info = json.loads(f.read())
    for key, value in layer_info.items():
        layer_info[key]['min_z'] -= size
        layer_info[key]['max_z'] += size
        print('{}, total z range = {} ({}-{})'.format(key, layer_info[key]['max_z'] - layer_info[key]['min_z'] + 1,
                                                      layer_info[key]['min_z'], layer_info[key]['max_z']))

    target_file = './data/all/gc_layers_reduced_mod{}.json'.format(size)
    with open(target_file, 'w') as f:
        f.write(json.dumps(layer_info))


def check(data_dir='./data/all/images', size=10):
    output_dir = './data/all/layer_check'
    target_file = './data/all/gc_layers_mod{}.json'.format(size) if size > 0 else './data/all/gc_layers_all.json'
    with open(target_file, 'r') as f:
        layer_info = json.loads(f.read())
    for filename in os.listdir(data_dir):
        if filename.endswith('.tif'):
            # load label coordinates
            file_path = os.path.join(data_dir, filename)
            file_prefix = filename[:-4]

            # load this volume
            volume_ori = skio.imread(file_path).astype(np.float64)  # volume axis: y, z, x
            volume_ori = np.transpose(volume_ori, axes=(1, 0, 2))  # volume axis: z, y, x

            min_z = layer_info[file_prefix]['min_z']
            max_z = layer_info[file_prefix]['max_z'] + 1
            volume = volume_ori[min_z: max_z, :, :]

            # add gaussian noise
            volume += np.random.normal(0, 1.5, size=volume.shape)
            # determine mean and std
            this_mean = np.mean(volume)
            this_std = np.std(volume)
            # normalize input volume
            volume = normalize_volume(volume, mean=this_mean, std=this_std)
            # random rotation
            volume, _, _ = random_rotate3d_xy(volume)
            # random flip
            volume, _, _ = random_flip(volume, axis=[0, 1, 2])

            volume = (volume - np.min(volume)) / (np.max(volume) - np.min(volume)) * 255
            volume = volume.astype(np.int8)

            this_output_dir = os.path.join(output_dir, file_prefix)
            if not os.path.exists(this_output_dir):
                os.makedirs(this_output_dir)

            for z_idx in np.arange(volume.shape[0]):
                this_slice = volume[z_idx]
                im = Image.fromarray(this_slice)
                im = im.convert('RGB')
                im.save(os.path.join(this_output_dir, '{}.png'.format(z_idx)))


if __name__ == '__main__':
    expand_size = 0
    # main()
    # modify_layer_region(size=expand_size)
    check(size=expand_size)
