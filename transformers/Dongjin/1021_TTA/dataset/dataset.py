from PIL import Image
from collections import defaultdict
from datasets import Dataset
import os
import albumentations as A
import numpy as np
import torch

def COCO2dataset(data_dir_path, coco, indices=False):
    # COCO 데이터를 timm datasets 형태로 변환
    # dataset으로 변환할 indices가 입력되지 않으면, COCO 데이터에 있는 모든 데이터를 사용함 

    if indices==False:
        indices = range(len(coco.getImgIds()))
    
    image_ids = coco.getImgIds()
    dictions = []

    for index in indices:
        image_id = image_ids[index]
        image_info = coco.loadImgs(image_id)[0]
        anns_ids = coco.getAnnIds(imgIds=image_id)
        anns = coco.loadAnns(anns_ids)

        image = Image.open(os.path.join(data_dir_path, image_info['file_name']))

        diction = {}
        diction['image_id'] = image_id
        diction['image'] = image
        diction['width'] = image_info['width']
        diction['height'] = image_info['height']

        objects = defaultdict(list)
        for ann in anns:
            objects['id'].append(ann['id'])
            objects['area'].append(ann['area'])
            objects['bbox'].append(ann['bbox'])
            objects['category'].append(ann['category_id'])

        diction['objects'] = dict(objects)
        dictions.append(diction)
    
    dataset = Dataset.from_list(dictions)
    return dataset


def get_transforms():
    train_augment_and_transform = A.Compose(
        [
            A.Perspective(p=0.1),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.HueSaturationValue(p=0.1),
        ],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True, min_area=25),
    )

    validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

    return train_augment_and_transform, validation_transform


def augment_and_transform_batch(examples, transform, image_processor, return_pixel_mask=False):
    """Apply augmentations and format annotations in COCO format for object detection task"""
    images = []
    annotations = []

    for image_id, image, objects in zip(examples["image_id"], examples["image"], examples["objects"]):
        image = np.array(image.convert("RGB"))

        # apply augmentations
        output = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
        images.append(output["image"])

        # format annotations in COCO format
        formatted_annotations = format_image_annotations_as_coco(
            image_id, output["category"], objects["area"], output["bboxes"]
        )
        annotations.append(formatted_annotations)

    # Apply the image processor transformations: resizing, rescaling, normalization
    result = image_processor(images=images, annotations=annotations, return_tensors="pt")
    result['images'] = images
    
    if not return_pixel_mask:
        result.pop("pixel_mask", None)

    return result


def format_image_annotations_as_coco(image_id, categories, areas, bboxes):
    """Format one set of image annotations to the COCO format

    Args:
        image_id (str): image id. e.g. "0001"
        categories (List[int]): list of categories/class labels corresponding to provided bounding boxes
        areas (List[float]): list of corresponding areas to provided bounding boxes
        bboxes (List[Tuple[float]]): list of bounding boxes provided in COCO format
            ([center_x, center_y, width, height] in absolute coordinates)

    Returns:
        dict: {
            "image_id": image id,
            "annotations": list of formatted annotations
        }
    """
    annotations = []
    for category, area, bbox in zip(categories, areas, bboxes):
        formatted_annotation = {
            "image_id": image_id,
            "category_id": category,
            "iscrowd": 0,
            "area": area,
            "bbox": list(bbox),
        }
        annotations.append(formatted_annotation)

    return {
        "image_id": image_id,
        "annotations": annotations,
    }



def collate_fn(batch):
    data = {}
    data["pixel_values"] = torch.stack([x["pixel_values"] for x in batch])
    data["labels"] = [x["labels"] for x in batch]
    if "pixel_mask" in batch[0]:
        data["pixel_mask"] = torch.stack([x["pixel_mask"] for x in batch])
    return data
