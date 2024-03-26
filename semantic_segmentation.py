import argparse
import itertools
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
import json
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


def render_segmentation(segmentation_logits, dataset):
    colormap = DATASET_COLORMAPS[dataset]
    colormap_array = np.array(colormap, dtype=np.uint8)
    segmentation_values = colormap_array[segmentation_logits + 1]
    return Image.fromarray(segmentation_values)


def calculate_iou(a, b, class_index):
    # 首先将a和b按照0/1的方式量化
    epsilon = 1e-5
    if class_index==-1:
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
    folder_list = [folder for folder in os.listdir(args.input_folder) if not folder.endswith('.json')]
    folder_list.sort(key=lambda x: int(x))
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
    for i, folder in enumerate(folder_list):
        print(f"Processing {i + 1}/{len(folder_list)}: {folder}")
        os.makedirs(os.path.join(args.output_folder, folder), exist_ok=True)
        image_list = [img for img in os.listdir(os.path.join(args.input_folder, folder)) if img.endswith('.jpg')]
        for i in range(len(image_list)):
            image_list[i] = image_list[i].split('.')[0].split('_')[0]
        random_seed_list = set(image_list)
        random_seed_iou_dict={}
        for random_seed in random_seed_list:
            original_image = []
            for index in ['0', '1']:
                image = Image.open(os.path.join(args.input_folder, folder, random_seed + '_' + index + '.jpg'))
                # in RGB order
                width, height = image.size
                new_height = int(height * 0.75)
                crop_area = (0, 0, width, new_height)
                image = image.crop(crop_area)
                # image.save(os.path.join(args.output_folder, folder, random_seed + '_' + index + '.jpg'))
                # print(f"saved {os.path.join(args.output_folder, folder, random_seed + '_' + index + '.jpg')}")
                array = np.array(image)[:, :, ::-1]  # BGR
                original_image.append(array)
            segmentation_output = []
            for index, img in enumerate(original_image):
                segmentation_logits = inference_segmentor(model, img)[0]  # (h,w)
                mask = np.isin(segmentation_logits, VEHICLE_LOGITS_TENSOR)
                segmentation_logits[~mask] = -1

                segmented_image = render_segmentation(segmentation_logits, "ade20k")
                # display(segmented_image)
                segmented_image.save(
                    os.path.join(args.output_folder, folder, random_seed + '_' + str(index) + '_seg.jpg'))
                print(f"saved {os.path.join(args.output_folder, folder, random_seed + '_' + str(index) + '_seg.jpg')}")
                segmentation_output.append(segmentation_logits.reshape(1, -1))

            # calculate iou of each class
            iou_res = {}
            iou_res['all'] = calculate_iou(segmentation_output[0], segmentation_output[1], -1)
            for class_index in VEHICLE_LOGITS_LIST:
                iou_res[VEHICLE_LOGIS_CLASS_MAPPING[class_index]] = calculate_iou(segmentation_output[0],
                                                                                  segmentation_output[1], class_index)
            random_seed_iou_dict[random_seed] = iou_res
        with open(os.path.join(args.output_folder, folder, 'iou.json'), 'w') as f:
            json.dump(random_seed_iou_dict, f,indent=4)
