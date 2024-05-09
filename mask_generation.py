import argparse
import itertools
import json
import math
import mmcv
import numpy as np
import sys
import torch
import torch.nn.functional as F
import urllib
import urllib
from PIL import Image
from functools import partial
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor, inference_segmentor

import dinov2.eval.segmentation.models
import dinov2.eval.segmentation_m2f.models.segmentors
from dinov2.models.vision_transformer import vit_base

sys.path.append('./')
import dinov2.eval.segmentation.utils.colormaps as colormaps
from torch.hub import set_dir
import os
import shutil

set_dir('./checkpoints/hub')

DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}
ADE20K_CLASS_NAMES = colormaps.ADE20K_CLASS_NAMES
VEHICLE_TARGET = ['car;auto;automobile;machine;motorcar',
                  'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle',
                  'truck;motortruck', 'van',
                  'minibike;motorbike', 'bicycle;bike;wheel;cycle']
VEHICLE_INDEX = ADE20K_CLASS_NAMES.index('car;auto;automobile;machine;motorcar')
# VEHICLE_COLOR = colormaps.ADE20K_COLORMAP[VEHICLE_INDEX]
VEHICLE_LOGITS_LIST = [ADE20K_CLASS_NAMES.index(target) - 1 for target in VEHICLE_TARGET]
VEHICLE_LOGITS_TENSOR = np.array(VEHICLE_LOGITS_LIST, dtype=np.int64)
print(f"target vehicle index tensor: {VEHICLE_LOGITS_TENSOR}")
VEHICLE_LOGIS_CLASS_MAPPING = {
    20: 'car',
    80: 'bus',
    83: 'truck',
    102: 'van',
    116: 'minibike',
    127: 'bicycle'
}


# target vehicle index tensor: [ 20  80  83 102 116 127], car,bus,truck,van,minibike,bicycle


# for i in range(len(ADE20K_CLASS_NAMES)):
#     if ADE20K_CLASS_NAMES[i] in ['car;auto;automobile;machine;motorcar',
#                                  'bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle',
#                                  'truck;motortruck', 'van',
#                                  'minibike;motorbike', 'bicycle;bike;wheel;cycle']:
#         DATASET_COLORMAPS['ade20k'][i] = VEHICLE_COLOR


def load_image_from_url(url: str) -> Image:
    with urllib.request.urlopen(url) as f:
        return Image.open(f).convert("RGB")


def load_config_from_url(url: str) -> str:
    with urllib.request.urlopen(url) as f:
        return f.read().decode()


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


def create_segmenter(cfg, backbone_model):
    model = init_segmentor(cfg)
    model.backbone.forward = partial(
        backbone_model.get_intermediate_layers,
        n=cfg.model.backbone.out_indices,
        reshape=True,
    )
    if hasattr(backbone_model, "patch_size"):
        model.backbone.register_forward_pre_hook(lambda _, x: CenterPadding(backbone_model.patch_size)(x[0]))
    model.init_weights()
    return model


def render_segmentation(segmentation_logits, dataset, white_and_black=False):
    if not white_and_black:
        colormap = DATASET_COLORMAPS[dataset]
    else:
        colormap = [(0, 0, 0),(255, 255, 255)]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)


def calculate_iou(a, b, class_index):
    # 首先将a和b按照0/1的方式量化
    epsilon = 1e-5
    if class_index == -1:
        a = (a != -1).astype(int)
        b = (b != -1).astype(int)
    else:
        a = (a == class_index).astype(int)
        b = (b == class_index).astype(int)

    # 计算交集(intersection)
    intersection = np.logical_and(a, b)
    intersection = np.sum(intersection)

    # 计算并集(union)
    union = np.logical_or(a, b)
    union = np.sum(union)

    # 计算IoU
    iou = intersection / (union + epsilon)

    return iou


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Use DinoV2 to filter image pair")
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder to be filtered")
    parser.add_argument("--output_folder", default=None, type=str, help="Path to the output folder")
    args = parser.parse_args()
    if args.output_folder is None:
        if args.input_folder.endswith('/'):
            args.output_folder = args.input_folder[:-1] + '_filtered'
        else:
            args.output_folder = args.input_folder + '_filtered'
    os.makedirs(args.output_folder, exist_ok=True)
    image_list = [image for image in os.listdir(args.input_folder) if not image.endswith('.json')]
    BACKBONE_SIZE = "giant"  # in ("small", "base", "large" or "giant")
    backbone_archs = {
        "small": "vits14",
        "base": "vitb14",
        "large": "vitl14",
        "giant": "vitg14",
    }
    backbone_arch = backbone_archs[BACKBONE_SIZE]
    backbone_name = f"dinov2_{backbone_arch}"
    print(f"loading {backbone_name}...")
    backbone_model = torch.hub.load("/home/turing/cfs_cz/finn/codes/dinov2/checkpoints", backbone_name, source="local",
                                    pretrained=False, force_reload=True)
    print(f"loaded {backbone_name}")
    backbone_model.eval()
    backbone_model.cuda()
    cfg = mmcv.Config.fromfile("checkpoints/dinov2_vitg14_ade20k_m2f_config.py")
    print(f"loading dinov2_vitg14_ade20k_m2f...")
    model = init_segmentor(cfg)
    load_checkpoint(model, "checkpoints/dinov2_vitg14_ade20k_m2f.pth", map_location="cpu")
    print(f"loaded!")
    model.cuda()
    model.eval()
    image_list=image_list[:10]
    for i, image_file in enumerate(image_list):
        image = Image.open(os.path.join(args.input_folder, image_file)).resize((512, 512))
        # in RGB order
        # image.save(os.path.join(args.output_folder, folder, random_seed + '_' + index + '.jpg'))
        # print(f"saved {os.path.join(args.output_folder, folder, random_seed + '_' + index + '.jpg')}")
        array = np.array(image)[:, :, ::-1]  # BGR
        segmentation_logits = inference_segmentor(model, array)[0]  # (h,w)
        mask = np.isin(segmentation_logits, VEHICLE_LOGITS_TENSOR)
        # set mask to 0, others to 1
        segmentation_logits[mask] = 0 # in render, 0 + 1 = 1 -> 白色 -> 不变
        segmentation_logits[~mask] = -1 # in render, -1 + 1 =0 -> 黑色 -> 变
        segmentation_logits[int(512*0.75):,:,]=-1
        segmented_image = render_segmentation(segmentation_logits, "ade20k", white_and_black=True)
        # display(segmented_image)
        segmented_image.save(
            os.path.join(args.output_folder, image_file))
