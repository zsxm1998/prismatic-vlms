"""
Utilities for bounding box manipulation and GIoU.
"""
import torch
from torchvision.ops.boxes import box_area
import re

from prismatic.constants import DEFAULT_BBOX_TOKEN


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


# modified from torchvision to also return the union
def box_iou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    union = area1[:, None] + area2 - inter

    iou = inter / (union + 1e-12)
    return iou, union


def generalized_box_iou(boxes1, boxes2):
    """
    Generalized IoU from https://giou.stanford.edu/

    The boxes should be in [x0, y0, x1, y1] format

    Returns a [N, M] pairwise matrix, where N = len(boxes1)
    and M = len(boxes2)
    """
    # degenerate boxes gives inf / nan results
    # so do an early check
    assert (boxes1[:, 2:] >= boxes1[:, :2]).all()
    assert (boxes2[:, 2:] >= boxes2[:, :2]).all()
    iou, union = box_iou(boxes1, boxes2)

    lt = torch.min(boxes1[:, None, :2], boxes2[:, :2])
    rb = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])

    wh = (rb - lt).clamp(min=0)  # [N,M,2]
    area = wh[:, :, 0] * wh[:, :, 1]

    return iou - (area - union) / (area + 1e-12)


def masks_to_boxes(masks):
    """Compute the bounding boxes around the provided masks

    The masks should be in format [N, H, W] where N is the number of masks, (H, W) are the spatial dimensions.

    Returns a [N, 4] tensors, with the boxes in xyxy format
    """
    if masks.numel() == 0:
        return torch.zeros((0, 4), device=masks.device)

    h, w = masks.shape[-2:]

    y = torch.arange(0, h, dtype=torch.float)
    x = torch.arange(0, w, dtype=torch.float)
    y, x = torch.meshgrid(y, x)

    x_mask = (masks * x.unsqueeze(0))
    x_max = x_mask.flatten(1).max(-1)[0]
    x_min = x_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    y_mask = (masks * y.unsqueeze(0))
    y_max = y_mask.flatten(1).max(-1)[0]
    y_min = y_mask.masked_fill(~(masks.bool()), 1e8).flatten(1).min(-1)[0]

    return torch.stack([x_min, y_min, x_max, y_max], 1)


def extract_and_replace_bboxes(input_str, bbox_tag="bbox", replace_with=DEFAULT_BBOX_TOKEN):
    # 定义正则表达式，匹配指定标签的bounding box
    pattern = re.compile(rf"<{bbox_tag}>(.*?)</{bbox_tag}>")
    
    # 提取所有bounding box
    bbox_strs = pattern.findall(input_str)
    
    # 将字符串形式的bounding box转换为浮点数列表
    bboxes = [list(map(float, bbox_str.split(','))) for bbox_str in bbox_strs]
    
    # 替换bounding box为指定字符串
    output_str = pattern.sub(replace_with, input_str)
    
    return output_str, bboxes


def restore_bboxes(output_str, bboxes, bbox_tag="bbox", replace_with=DEFAULT_BBOX_TOKEN):
    # 将tensor转换为字符串列表
    bbox_strs = [", ".join(map(lambda x: f"{x:.3f}", bbox.tolist())) for bbox in bboxes]
    # 用于替换的正则表达式
    pattern = re.compile(re.escape(replace_with))
    # 用于查找的位置指针
    pos = 0
    # 逐个替换占位符为bounding box字符串，从左到右
    for bbox_str in bbox_strs:
        match = pattern.search(output_str, pos)
        if match:
            start, end = match.span()
            output_str = output_str[:start] + f"<{bbox_tag}>{bbox_str}</{bbox_tag}>" + output_str[end:]
            pos = start + len(f"<{bbox_tag}>{bbox_str}</{bbox_tag}>")  # 更新查找位置
    return output_str