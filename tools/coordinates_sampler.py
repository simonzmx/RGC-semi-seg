import os
import skimage.io as skio
import scipy.io as scio
import numpy as np
import json
from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
from matplotlib.patches import Circle


def sample_pos_neg_coordinates(ratios, root_path='data/all'):
    img_path = os.path.join(root_path, 'images')
    label_path = os.path.join(root_path, 'labels')
    with open('data/all/gc_layers_all.json', 'r') as f:
        layer_info = json.loads(f.read())

    count_dict = {ratio: {'pos': 0, 'neg': 0} for ratio in ratios}
    for img_idx, img_file in enumerate(os.listdir(img_path)):
        if img_file.endswith('.tif'):
            print('Sampling positive and negative coordinates for image {} ...'.format(img_file))
            np.random.seed(42)

            img_filename = img_file.split(os.sep)[-1][:-4]
            # load this volume
            volume = skio.imread(os.path.join(img_path, img_file))  # volume axis: y, z, x
            volume = np.transpose(volume, axes=(1, 0, 2))  # volume axis: z, y, x

            # load label coordinates
            label_file_path = os.path.join(label_path, img_file.split(os.sep)[-1][:-4] + '_marker.mat')
            mat_content = scio.loadmat(label_file_path)
            gc_coords = np.rint(mat_content['temp_marker']).astype(np.int32)  # coordinates axis: z, x, y
            gc_coords[:, [1, 2]] = gc_coords[:, [2, 1]]  # coordinates axis: z, y, x
            gc_coords -= 1  # original coordinates start from 1, make them start from 0

            if img_filename in layer_info:
                min_z = max(layer_info[img_filename]['min_z'], 0)
                max_z = min(layer_info[img_filename]['max_z'] + 1, volume.shape[0])
            else:
                min_z = max(np.min(gc_coords[:, 0]), 0)
                max_z = min(np.max(gc_coords[:, 0]) + 1, volume.shape[0])

            # get available positive coordinates, within the volume body
            selected_pos_coords = []
            for pos_coord in gc_coords:
                if 0 <= int(pos_coord[0]) < volume.shape[0] and 0 <= int(pos_coord[1]) < volume.shape[1] \
                        and 0 <= int(pos_coord[2]) < volume.shape[2]:
                    selected_pos_coords.append(pos_coord.astype(np.int32))
                else:
                    print(f'{volume.shape} --- {pos_coord}')

            # get all possible negative coordinates via voronoi method
            vor_coords = Voronoi(selected_pos_coords)
            # get available negative coordinates, within the cropped volume body
            selected_neg_coords_pre = []
            for neg_coord in vor_coords.vertices:
                if min_z <= int(neg_coord[0]) < max_z and 0 <= int(neg_coord[1]) < volume.shape[1] \
                        and 0 <= int(neg_coord[2]) < volume.shape[2]:
                    selected_neg_coords_pre.append(neg_coord.astype(np.int32))
            # remove nearby negative coordinates
            selected_neg_coords = []
            for idx1, coord1 in enumerate(selected_neg_coords_pre):
                avl = True
                for idx2, coord2 in enumerate(selected_neg_coords_pre[idx1 + 1:]):
                    if np.linalg.norm(coord1 - coord2) <= 5:
                        avl = False
                        break
                for idx2, coord2 in enumerate(selected_pos_coords):
                    if np.linalg.norm(coord1 - coord2) <= 5:
                        avl = False
                        break
                if avl:
                    selected_neg_coords.append(coord1)

            np.random.shuffle(selected_pos_coords)
            np.random.shuffle(selected_neg_coords)

            for ratio in ratios:
                target_path = os.path.join(root_path, 'labels_{:.2f}'.format(ratio))
                if not os.path.exists(target_path):
                    os.makedirs(target_path)

                # randomly select some positive/negative coordinates according to the label usage ratio
                selected_pos_k = np.ceil(len(selected_pos_coords) * ratio).astype(np.int32)
                this_selected_pos_coords = np.concatenate((selected_pos_coords[:selected_pos_k],
                                                           np.ones([selected_pos_k, 1])), axis=1)
                selected_neg_k = min(selected_pos_k, np.ceil(len(selected_neg_coords) * ratio).astype(np.int32))
                this_selected_neg_coords = np.concatenate((selected_neg_coords[:selected_neg_k],
                                                           np.zeros([selected_neg_k, 1])), axis=1)

                all_coords = np.concatenate((this_selected_pos_coords, this_selected_neg_coords))
                np.save(os.path.join(target_path, '{}.npy'.format(img_file.split(os.sep)[-1][:-4])),
                        all_coords)

                if 'IU' in img_file:  # and 'Glaucoma' in img_file:
                    count_dict[ratio]['pos'] += len(this_selected_pos_coords)
                    count_dict[ratio]['neg'] += len(this_selected_neg_coords)
    print(count_dict)


def check_coordinate_samples(root_path='data/all', ratio=0.5):
    img_path = os.path.join(root_path, 'images')
    sample_path = os.path.join(root_path, 'labels_{:.2f}'.format(ratio))
    with open('data/all/gc_layers_all.json', 'r') as f:
        layer_info = json.loads(f.read())

    for img_idx, img_file in enumerate(os.listdir(img_path)):
        if img_file.endswith('.tif') and img_file.startswith('IU'):
            print('Checking on image {} ...'.format(img_file))
            img_filename = img_file.split(os.sep)[-1][:-4]
            # load this volume
            volume = skio.imread(os.path.join(img_path, img_file))  # volume axis: y, z, x
            volume = np.transpose(volume, axes=(1, 0, 2))  # volume axis: z, y, x

            all_coords = np.load(os.path.join(sample_path, '{}.npy'.format(img_file.split(os.sep)[-1][:-4])))
            min_z = int(np.min(all_coords[:, 0]))
            max_z = int(np.max(all_coords[:, 0]))

            save_dir = os.path.join(sample_path, img_file[:-4])
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            for z_idx in range(min_z, max_z + 1):
                fig, axs = plt.subplots(1, figsize=(16, 20))

                # show original slice
                axs.imshow(volume[z_idx], cmap='gray')
                axs.set_title(f"z={z_idx}")
                axs.set_axis_off()

                for idx, coord in enumerate(all_coords):
                    if coord[0] == z_idx:
                        if coord[3] == 1:
                            # Pos - cyan
                            axs.add_patch(Circle((coord[2], coord[1]), radius=2, color='cyan'))
                        else:
                            # Neg - red
                            axs.add_patch(Circle((coord[2], coord[1]), radius=2, color='red'))

                fig.savefig(os.path.join(save_dir, f'{z_idx:04d}.png'), bbox_inches='tight')
                plt.close()


def check_overlap(root_path='data/all'):
    img_path = os.path.join(root_path, 'images')
    ratios = [0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    for idx, r in enumerate(ratios[:-1]):
        print('Checking between ratio={:.2f} and ratio={:.2f} ...'.format(ratios[idx], ratios[idx + 1]))
        sample_path = os.path.join(root_path, 'labels_{:.2f}'.format(ratios[idx]))
        sample_path_f = os.path.join(root_path, 'labels_{:.2f}'.format(ratios[idx + 1]))

        for img_idx, img_file in enumerate(os.listdir(img_path)):
            if img_file.endswith('.tif'):
                all_coords = np.load(os.path.join(sample_path, '{}.npy'.format(img_file.split(os.sep)[-1][:-4])))
                all_coords_f = np.load(os.path.join(sample_path_f, '{}.npy'.format(img_file.split(os.sep)[-1][:-4])))

                for row in all_coords:
                    assert any(np.equal(all_coords_f, row).all(1))


if __name__ == '__main__':
    ratios = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.01]
    sample_pos_neg_coordinates(ratios)
    check_coordinate_samples(ratio=1.0)
    check_overlap()
