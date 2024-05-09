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
import collections
from tqdm import tqdm
import random

set_dir('./checkpoints/hub')

DATASET_COLORMAPS = {
    "ade20k": colormaps.ADE20K_COLORMAP,
    "voc2012": colormaps.VOC2012_COLORMAP,
}
ADE20K_CLASS_NAMES = colormaps.ADE20K_CLASS_NAMES
TARGET_CLASS_NAMES_LOGITS_INDEX = {
    "car;auto;automobile;machine;motorcar": None,
    "truck;motortruck": None,
    "bus;autobus;coach;charabanc;double-decker;jitney;motorbus;motorcoach;omnibus;passenger;vehicle": None,
    "van": None,
    "minibike;motorbike": None,
    "bicycle;bike;wheel;cycle": None,
    "building;edifice": None,
    "streetlight;street;lamp": None,
    "sky": None,
    "tree": None,
    "road;route": None,
    "sidewalk;pavement": None,
    "person;individual;someone;somebody;mortal;soul": None,
    "bridge;span": None,
    "mountain;mount": None,
    "railing;rail": None,
    "signboard;sign": None,
    "traffic;light;traffic;signal;stoplight": None,
    "grass": None,
    "plant;flora;plant;life": None,
    "fence;fencing": None,
}
for key, value in TARGET_CLASS_NAMES_LOGITS_INDEX.items():
    TARGET_CLASS_NAMES_LOGITS_INDEX[key] = ADE20K_CLASS_NAMES.index(key) - 1
print(TARGET_CLASS_NAMES_LOGITS_INDEX)


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
    parser.add_argument("--input_folder", type=str, required=True, help="Path to the folder with images and jsonl")
    args = parser.parse_args()

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
    sem_result_dict = collections.defaultdict(int)
    json_file_path = os.path.join(args.input_folder, 'metadata.jsonl')
    data_to_process = []
    with open(json_file_path, 'r') as file:
        for line in file:
            data = json.loads(line)
            data_to_process.append(data)
    for i, data in enumerate(tqdm(data_to_process)):
        image = Image.open(os.path.join(args.input_folder, data['file_name']))
        array = np.array(image)[:, :, ::-1]  # BGR
        segmentation_logits = inference_segmentor(model, array)[0]  # (h,w)
        addition_semantic_label = ""
        logits = np.unique(segmentation_logits).tolist()
        for key, value in TARGET_CLASS_NAMES_LOGITS_INDEX.items():
            if value in logits:
                semantic_label = key.split(";")[0]
                addition_semantic_label += semantic_label
                addition_semantic_label += ","
        addition_semantic_label = addition_semantic_label[:-1]
        data['dino_semantic_label'] = addition_semantic_label
        # sentences = data['text'].split(',')
        # sentences.insert(1, addition_semantic_label)
        # data['text'] = '.'.join(','.join(sentences).split('.')[:-2])
        # data['text'] += '.'
    output_file = json_file_path.replace('.jsonl', '_dinov2_label.jsonl')
    with open(output_file, 'w') as f:
        for item in data_to_process:
            f.write(json.dumps(item) + "\n")
    print(f"Saved to {output_file}")
