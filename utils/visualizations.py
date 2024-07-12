import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from matplotlib.patches import Rectangle, Circle, RegularPolygon
from utils.measurements import measure_performance_pred_thresholded


def generate_precision_recall_curve(precisions, recalls):
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

    # plot
    fig, ax = plt.subplots()
    ax.plot(recalls, precisions, 'o-r')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_xticks(np.arange(0, 1.01, 0.2))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    return fig


def generate_model_prediction_map(ori_vol, label_vol, pred_vol, save_dir, img_folder, median_vol=None, max_vol=None):
    for z_idx in range(ori_vol.shape[0]):
        if not os.path.exists(os.path.join(save_dir, img_folder)):
            os.makedirs(os.path.join(save_dir, img_folder))

        if median_vol is not None or max_vol is not None:
            fig, axs = plt.subplots(3, 2, figsize=(16, 20))
        else:
            fig, axs = plt.subplots(2, 2, figsize=(16, 20))

        # show original slice
        axs[0, 0].imshow(ori_vol[z_idx], cmap='gray')
        axs[0, 0].set_title(f"original slice at z={z_idx}")

        # show ground-truth slice
        axs[0, 1].imshow(ori_vol[z_idx], cmap=plt.cm.gray, interpolation='nearest')
        axs[0, 1].imshow(label_vol[z_idx], cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear')
        axs[0, 1].set_title(f"original slice overlayed by ground-truth label at z={z_idx}")

        # show prediction slice
        axs[1, 0].imshow(pred_vol[z_idx], vmin=0.0, vmax=1.0)
        axs[1, 0].set_title(f"Predicted probability map at z={z_idx}")

        # show prediction slice
        axs[1, 1].imshow(pred_vol[z_idx], cmap=plt.cm.gray, interpolation='nearest')
        axs[1, 1].imshow(label_vol[z_idx], cmap=plt.cm.viridis, alpha=.9, interpolation='bilinear')
        axs[1, 1].set_title(f"Predicted probability map overlayed by gt label at z={z_idx}")

        if median_vol is not None:
            # show median filtered slice
            axs[2, 0].imshow(median_vol[z_idx], vmin=0.0, vmax=1.0)
            axs[2, 0].set_title(f"Median filtered probability map at z={z_idx}")
        if max_vol is not None:
            # show max filtered slice
            axs[2, 1].imshow(max_vol[z_idx])
            axs[2, 1].set_title(f"Max filtered probability map at z={z_idx}")

        fig.savefig(os.path.join(save_dir, img_folder, f'{z_idx:04d}.png'), bbox_inches='tight')
        plt.close()


def generate_evaluation_figures(volume, pred_coords, gt_coords, gt_matched, pred_matched, scores,
                                save_dir, img_folder, voxel_res, offset=5, scale_bar=50):
    img_folder = f'{img_folder}_off{offset}'
    if not os.path.exists(os.path.join(save_dir, img_folder)):
        os.makedirs(os.path.join(save_dir, img_folder))

    # text_kwargs = dict(ha='left', va='center', fontsize=48)
    voxel_off = int(voxel_res[0] * offset)
    bar_length = int(voxel_res[1] * scale_bar)
    for z_idx in range(volume.shape[0]):
        fig, axs = plt.subplots(1, figsize=(16, 20))

        # show original slice
        slice = volume[z_idx].astype(np.uint8)
        slice = auto_contrast(slice)  # auto contrast (similar to the one in ImageJ) for better visualization
        axs.imshow(slice, cmap='gray')
        # axs.set_title(f"original en face slice at z={z_idx}")

        for gt_idx, gt_coord in enumerate(gt_coords):
            if abs(gt_coord[0] - z_idx) <= voxel_off and gt_matched[gt_idx] == 0:
                # FN - red
                axs.add_patch(RegularPolygon((gt_coord[2], gt_coord[1]), numVertices=3, radius=3, color='red'))
                # axs.text(gt_coord[2], gt_coord[1], f'{gt_idx}', color='red', **text_kwargs)
        for pred_idx, pred_coord in enumerate(pred_coords):
            if abs(pred_coord[0] - z_idx) <= voxel_off and pred_matched[pred_idx] >= 0:
                if pred_matched[pred_idx] > 0:
                    # TP - cyan
                    axs.add_patch(RegularPolygon((pred_coord[2], pred_coord[1]), numVertices=3, radius=3, color='cyan'))
                    # axs.text(pred_coord[2], pred_coord[1], f'{pred_idx}-{int(pred_matched[pred_idx] - 1)}', color='cyan', **text_kwargs)
                else:
                    # FP - yellow
                    axs.add_patch(Circle((pred_coord[2], pred_coord[1]), radius=2, color='yellow'))
                    # axs.text(pred_coord[2], pred_coord[1], f'{pred_idx}', color='yellow', **text_kwargs)

        # scale bar
        sb_y = int(volume.shape[1] * 0.95)
        sb_x = int(volume.shape[2] * 0.05)
        plt.plot((sb_x, sb_x + bar_length), (sb_y, sb_y), '-', color='white', linewidth=5)

        # scores
        # axs.text(int(volume.shape[2] * 0.03), int(volume.shape[1] * 0.05), 'F1={:.3f}'.format(scores['f1s_vt'][-1]),
        #          color='white', **text_kwargs)
        # axs.text(int(volume.shape[2] * 0.03), int(volume.shape[1] * 0.10), 'AP={:.3f}'.format(scores['avg_pres'][-1]),
        #          color='white', **text_kwargs)

        plt.axis('off')
        fig.savefig(os.path.join(save_dir, img_folder, f'{z_idx:04d}.png'), bbox_inches='tight')
        plt.close()


def visualize_volume(volume, coord, patch_half, patch_size):
    # volume: z, y, x
    # plotly show
    X, Y, Z = np.mgrid[
              coord[2] - patch_half[2]: coord[2] - patch_half[2] + patch_size[2],
              coord[1] - patch_half[1]: coord[1] - patch_half[1] + patch_size[1],
              coord[0] - patch_half[0]: coord[0] - patch_half[0] + patch_size[0]]
    volume_trans = np.transpose(volume, axes=(2, 1, 0))
    fig = go.Figure(data=go.Volume(
        x=X.flatten(),
        y=Y.flatten(),
        z=Z.flatten(),
        value=volume_trans.flatten(),
        isomin=-1,
        isomax=1,
        opacity=0.1,  # needs to be small to see through all surfaces
        surface_count=20,  # needs to be a large number for good volume rendering
    ))
    fig.show()


def visualize_volume_mid_xy_yz_planes(full_volume, volume, coord, z_min, patch_half, patch_size, label):
    # volume: z, y, x
    x_mid = int(volume.shape[2] / 2)
    y_mid = int(volume.shape[1] / 2)
    z_mid = int(volume.shape[0] / 2)
    x_range = [coord[2] - patch_half[2], coord[2] - patch_half[2] + patch_size[2]]
    y_range = [coord[1] - patch_half[1], coord[1] - patch_half[1] + patch_size[1]]
    z_range = [coord[0] - patch_half[0], coord[0] - patch_half[0] + patch_size[0]]
    if label == 1:
        label = 'positive'
    elif label == 0:
        label = 'negative'
    else:
        label = 'None'

    fig, axes = plt.subplots(2, 3)
    fig.suptitle('{} sample: z,y,x coordinate ({}, {}, {})'.format(label, int(z_min + coord[0] + 1), coord[1], coord[2]))

    axes[0][0].set_title('xy plane')
    axes[0][0].set(xlabel='x', ylabel='y')
    axes[0][0].imshow(full_volume[coord[0], :, :],
                      vmin=full_volume.min(), vmax=full_volume.max())
    rect1 = Rectangle((x_range[0], y_range[0]), patch_size[2], patch_size[1], linewidth=1, edgecolor='r', facecolor='none')
    axes[0][0].add_patch(rect1)

    axes[0][1].set_title('yz plane')
    axes[0][1].set(xlabel='y', ylabel='z')
    axes[0][1].imshow(full_volume[:, :, coord[2]],
                      vmin=full_volume.min(), vmax=full_volume.max())
    rect2 = Rectangle((y_range[0], z_range[0]), patch_size[1], patch_size[0], linewidth=1, edgecolor='r',
                      facecolor='none')
    axes[0][1].add_patch(rect2)

    axes[0][2].set_title('xz plane')
    axes[0][2].set(xlabel='x', ylabel='z')
    axes[0][2].imshow(full_volume[:, coord[1], :],
                      vmin=full_volume.min(), vmax=full_volume.max())
    rect2 = Rectangle((x_range[0], z_range[0]), patch_size[2], patch_size[0], linewidth=1, edgecolor='r',
                      facecolor='none')
    axes[0][2].add_patch(rect2)

    axes[1][0].set_title('xy plane')
    axes[1][0].set(xlabel='x', ylabel='y')
    axes[1][0].imshow(volume[z_mid, :, :],
                      extent=[x_range[0], x_range[-1], y_range[-1], y_range[0]],
                      vmin=full_volume.min(), vmax=full_volume.max())

    axes[1][1].set_title('yz plane')
    axes[1][1].set(xlabel='y', ylabel='z')
    axes[1][1].imshow(volume[:, :, x_mid],
                      extent=[y_range[0], y_range[-1], z_range[-1], z_range[0]],
                      vmin=full_volume.min(), vmax=full_volume.max())

    axes[1][2].set_title('xz plane')
    axes[1][2].set(xlabel='x', ylabel='z')
    axes[1][2].imshow(volume[:, y_mid, :],
                      extent=[x_range[0], x_range[-1], z_range[-1], z_range[0]],
                      vmin=full_volume.min(), vmax=full_volume.max())

    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    plt.show()


def auto_contrast(img):
    # ImageJ auto contrast
    # https://forum.image.sc/t/macro-for-image-adjust-brightness-contrast-auto-button/37157/5
    im_type = img.dtype
    im_min = np.min(img)
    im_max = np.max(img)

    if len(img.shape) == 3 and img.shape[2] == 3:
        img = 0.3 * img[:, :, 2] + 0.59 * img[:, :, 1] + 0.11 * img[:, :, 0]
        img = img.astype(im_type)

    if im_type == np.uint8:
        hist_min = 0
        hist_max = 256
    elif im_type in (np.uint16, np.int32):
        hist_min = im_min
        hist_max = im_max
    else:
        raise NotImplementedError(f"Not implemented for dtype {im_type}")

    histogram = np.histogram(img, bins=256, range=(hist_min, hist_max))[0]
    bin_size = (hist_max - hist_min) / 256

    h, w = img.shape[:2]
    pixel_count = h * w
    limit = pixel_count / 10
    const_auto_threshold = 5000
    auto_threshold = 0

    auto_threshold = const_auto_threshold if auto_threshold <= 10 else auto_threshold / 2
    threshold = int(pixel_count / auto_threshold)

    i = -1
    found = False
    while not found and i <= 255:
        i += 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
    hmin = i
    found = False

    i = 256
    while not found and i > 0:
        i -= 1
        count = histogram[i]
        if count > limit:
            count = 0
        found = count > threshold
    hmax = i

    if hmax >= hmin:
        min_ = hist_min + hmin * bin_size
        max_ = hist_min + hmax * bin_size
        if min_ == max_:
            min_ = hist_min
            max_ = hist_max
    else:
        min_ = hist_min
        max_ = hist_max

    imr = (img - min_) / (max_ - min_) * 255
    # print(min_, max_, imr.min(), imr.max())
    imr = np.clip(imr, 0, 255)
    return imr
