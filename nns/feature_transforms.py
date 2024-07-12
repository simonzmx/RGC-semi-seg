import warnings
import torch
import random
import torch.nn.functional as F


def guided_masking(x, output, return_msk_context=True):
    if len(output.shape) == 4:
        masks_context = (output > 0).float().unsqueeze(1)
    else:
        masks_context = (output.argmax(1) > 0).float().unsqueeze(1)

    masks_context = F.interpolate(masks_context, size=x.shape[2:5], mode='nearest')

    if return_msk_context:
        masks = masks_context
    else:
        masks = (1 - masks_context)
    x_masked = masks * x
    return x_masked


def feature_perturbations(feature: torch.Tensor,
                          perturbation: str,
                          out_main_decoder: torch.Tensor = None):
    if perturbation == 'F-Noise':
        noise = torch.rand(feature.shape) * (-0.3 - 0.3) + 0.3
        noise = noise.to(feature.device)
        perturbed_feature = feature * noise + noise
    elif perturbation == 'F-Drop':
        threshold_drop = random.uniform(0.6, 0.9)
        normalized_feature = feature.clone()
        normalized_feature = normalized_feature.view(normalized_feature.shape[0], -1)
        normalized_feature -= normalized_feature.min(1, keepdim=True)[0]
        normalized_feature /= normalized_feature.max(1, keepdim=True)[0] + 1e-6
        normalized_feature = normalized_feature.view(feature.shape)
        mask_drop = (normalized_feature <= threshold_drop)
        perturbed_feature = feature * mask_drop
    elif perturbation == 'Obj-Mask':
        # TODO: uncertain about how this mask should be built
        out_resized = F.interpolate(out_main_decoder, size=feature.shape[2:5], mode='trilinear')
        out_max_index = out_resized.max(1, keepdim=True)[1]
        mask_object = (out_max_index == 1)
        perturbed_feature = feature * mask_object
    elif perturbation == 'Con-Mask':
        out_resized = F.interpolate(out_main_decoder, size=feature.shape[2:5], mode='trilinear')
        out_max_index = out_resized.max(1, keepdim=True)[1]
        mask_context = (out_max_index == 0)
        perturbed_feature = feature * mask_context
    elif perturbation == 'Spatial-Drop':
        spatial_dropout = torch.nn.Dropout(p=0.1)
        perturbed_feature = spatial_dropout(feature)
    else:
        # TODO: Guided Cutout (G-Cutout) perturbation
        warnings.warn(f'Unsupported feature perturbation operation: \"{perturbation}\". Identical feature used.')
        perturbed_feature = feature
    return perturbed_feature
