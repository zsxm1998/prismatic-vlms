import re
import numpy as np
import cv2
import torch
from torch import Tensor

from prismatic.constants import DEFAULT_MASK_TOKEN


def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    assert input.dim() == 3 or not reduce_batch_first

    sum_dim = (-1, -2) if input.dim() == 2 or not reduce_batch_first else (-1, -2, -3)

    inter = 2 * (input * target).sum(dim=sum_dim)
    sets_sum = input.sum(dim=sum_dim) + target.sum(dim=sum_dim)
    sets_sum = torch.where(sets_sum == 0, inter, sets_sum)

    dice = (inter + epsilon) / (sets_sum + epsilon)
    return dice.mean()


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon: float = 1e-6):
    # Average of Dice coefficient for all classes
    return dice_coeff(input.flatten(0, 1), target.flatten(0, 1), reduce_batch_first, epsilon)


def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)


def find_white_regions(image, max_vertices=50):
    """
    Find white regions in a binary image and return their bounding boxes and boundary polygons.
    This version adjusts epsilon dynamically to achieve the desired number of vertices for polygons.

    :param image: A single-channel binary image.
    :param max_vertices: Maximum number of vertices for the boundary polygon.
    :return: Two lists - one for bounding boxes and one for boundary polygons.
    """
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bounding_boxes = []
    boundary_polygons = []

    # Check if there are no contours found
    if not contours:
        return bounding_boxes, boundary_polygons  # Return empty lists if no contours found

    img_height, img_width = image.shape

    for contour in contours:
        # Compute the bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        bounding_boxes.append([round(x / img_width, 3), round(y / img_height, 3), 
                               round((x + w) / img_width, 3), round((y + h) / img_height, 3)])

        # Dynamically adjust epsilon to reduce the number of vertices
        epsilon = 0.001 * cv2.arcLength(contour, True)  # initial epsilon
        while True:
            approx_polygon = cv2.approxPolyDP(contour, epsilon, True)
            if len(approx_polygon) <= max_vertices:
                break
            epsilon *= 1.1  # Increase epsilon

        # Normalize the coordinates of the polygon points and add to the list
        polygon = [[round(point[0][0] / img_width, 3), round(point[0][1] / img_height, 3)] for point in approx_polygon]
        boundary_polygons.append(polygon)

    # Sorting the regions based on the top-left corner of bounding boxes
    sorted_combined = sorted(zip(bounding_boxes, boundary_polygons), key=lambda x: (x[0][0], x[0][1]))

    # Unzipping the sorted pairs
    bounding_boxes_sorted, boundary_polygons_sorted = map(list, zip(*sorted_combined)) if sorted_combined else ([], [])

    return bounding_boxes_sorted, boundary_polygons_sorted


def extract_and_replace_masks(contour_str, image_size, contour_tag='contour_list', polygon_tag='polygon', replacement_tag=DEFAULT_MASK_TOKEN):
    """
    输入： 
    - contour_str: 包含<contour_tag>和<polygon_tag>的字符串
    - image_size: 图像尺寸（宽, 高）
    - contour_tag: 标签用于包裹多边形的外部标签名称
    - polygon_tag: 标签用于包裹单个多边形的标签名称
    - replacement_tag: 用于替换contour_tag的标签
    
    返回：
    - new_contour_str: 替换了contour_tag为replacement_tag的字符串
    - masks: 包含每个contour_tag对应的掩码图像列表
    """
    # 定义正则表达式，提取多个contour_tag块
    contour_pattern = f"<{contour_tag}>.*?</{contour_tag}>"
    
    # 提取所有contour_tag块
    contour_blocks = re.findall(contour_pattern, contour_str, re.DOTALL)
    
    # 定义正则表达式，提取多边形中的坐标，兼容圆括号和方括号
    polygon_pattern = f"<{polygon_tag}>[\\(\\[].*?[\\)\\]]</{polygon_tag}>"
    
    # 初始化空的掩码图像列表
    masks = []

    # 替换后的字符串初始化为原始的contour_str
    new_contour_str = contour_str

    # 对每个contour_tag块进行处理
    for block in contour_blocks:
        # 初始化一个纯黑图像
        mask = np.zeros((image_size[1], image_size[0]), dtype=np.uint8)
        
        # 提取该块中的所有多边形
        polygons = re.findall(polygon_pattern, block)
        
        # 解析并处理每个多边形
        for polygon in polygons:
            # 提取坐标对
            coords = re.findall(r'\(?(\d*\.\d+|\d+),\s*(\d*\.\d+|\d+)\)?', polygon)
            
            # 转换为实际坐标值
            points = [(int(float(x) * image_size[0]), int(float(y) * image_size[1])) for x, y in coords]
            
            # 转换为符合OpenCV格式的 numpy 数组
            points_array = np.array([points], dtype=np.int32)
            
            # 在mask图像上填充绘制多边形
            cv2.fillPoly(mask, points_array, 255)
        
        # 将该块的mask添加到结果列表中
        masks.append(mask)
        
        # 替换该块为新的replacement_tag
        new_contour_str = new_contour_str.replace(block, replacement_tag)
    
    return new_contour_str, masks


def restore_masks(new_contour_str, masks, replacement_tag=DEFAULT_MASK_TOKEN, contour_tag='contour_list', polygon_tag='polygon'):
    """
    输入：
    - new_contour_str: 包含replacement_tag标签的字符串
    - masks: 掩码图像列表
    - replacement_tag: 标签用于替换回原始的contour_tag
    - contour_tag: 用于还原的多边形外部标签名称
    - polygon_tag: 用于还原的多边形标签名称
    
    返回：
    - contour_str: 还原后的包含<contour_tag>和<polygon_tag>的字符串
    """
    # 初始化contour_str为新的输入字符串
    contour_str = new_contour_str
    
    # 使用cv2.findContours从mask中提取轮廓
    for mask in masks:
        # 提取轮廓
        _, contours = find_white_regions(mask)
        
        # 初始化一个contour_list块
        contour_block = f"<{contour_tag}>"
        
        # 处理每个轮廓
        for contour in contours:
            # 初始化一个polygon块
            polygon_block = f"<{polygon_tag}>"
            
            # 将轮廓坐标转换为相对坐标（比例形式）
            for x_rel, y_rel in contour:
                # 将坐标加入polygon块
                polygon_block += f"[{x_rel:.3f}, {y_rel:.3f}], "
            
            # 去掉最后一个逗号和空格
            polygon_block = polygon_block.rstrip(", ")
            polygon_block += f"</{polygon_tag}>"
            
            # 将polygon块加入contour_block
            contour_block += polygon_block
        
        # 完成contour_block
        contour_block += f"</{contour_tag}>"
        
        # 将 <replacement_tag> 替换为还原的 contour_block
        contour_str = contour_str.replace(replacement_tag, contour_block, 1)
    
    return contour_str
