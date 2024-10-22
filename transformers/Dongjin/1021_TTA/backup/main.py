# %%
import os
from pycocotools.coco import COCO
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer
from functools import partial
import argparse
import utils
import dataset
import train_eval
from glob import glob
import torch
from tqdm import tqdm
import albumentations as A


# %%

model_path = '/data/ephemeral/home/Dongjin/level2-objectdetection-cv-07/transformers/Dongjin/1011_model_search/result/1015/jozhang97/deta-swin-large_3_img_size_720'
coco_dir_path = '/data/ephemeral/home/Dongjin/level2-objectdetection-cv-07/Split_data'

# model_path = '../../../transformers/Dongjin/1011_model_search/result/1015/jozhang97/deta-swin-large_3_img_size_720'
# coco_dir_path = '../../../Split_data'
device = 'cuda'

checkpoint_path = utils.find_checkpoint_path(model_path)
run_name = os.path.split(model_path)[-1]
json_path = os.path.join(model_path, run_name + '.json')

conf = utils.read_json(json_path)
conf['coco_dir_path'] = coco_dir_path
image_processor = AutoImageProcessor.from_pretrained(checkpoint_path)

valid_info_path = os.path.join(conf['coco_dir_path'], conf['valid_info_name'])
coco_valid = COCO(valid_info_path)

valid = dataset.COCO2dataset(conf['data_dir_path'], coco_valid)

validation_transform = A.Compose(
        [A.NoOp()],
        bbox_params=A.BboxParams(format="coco", label_fields=["category"], clip=True),
    )

validation_transform_batch = partial(
    dataset.augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
)

valid = valid.with_transform(validation_transform_batch)

model = AutoModelForObjectDetection.from_pretrained(checkpoint_path)
model.to(device)


# %%
def test_eval(conf, model, image_processor, coco_data, data, result_path):
    batch_size = 8

    image_names = []
    prediction_strings = []
    batch_indices = get_batch_indices(batch_size, len(data))

    for batch_index in tqdm(batch_indices):
        batch = data[batch_index]
        image_name, prediction_string = get_predictions(batch, coco_data, model, image_processor)
        
        image_names.extend(image_name)
        prediction_strings.extend(prediction_string)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = image_names
    submission.to_csv(result_path, index=None)


def get_batch_indices(batch_size, n):
    i = 0
    batch_index = []
    batch_indices = []

    while True:
        if (i == n): 
            batch_indices.append(batch_index)
            break

        batch_index.append(i)

        if (len(batch_index) % batch_size == 0):
            batch_indices.append(batch_index)
            batch_index = []
            
        i += 1  

    return batch_indices
        

def get_predictions(batch, coco_data, model, image_processor):
    threshold = 0.05
    device = "cuda"

    with torch.no_grad():
        images = batch['image']
        image_ids = batch['image_id']
        image_infos = coco_data.loadImgs(image_ids)

        inputs = image_processor(images=images, return_tensors="pt")
        outputs = model(**inputs.to(device))
        target_sizes = [image.size for image in images]
        results = image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)
        
        image_names = [image_info['file_name'] for image_info in image_infos]
        prediction_strings = []

        for result in results:
            _, indices = torch.sort(result['scores'], descending=True)
            result["scores"] = result["scores"][indices].detach().cpu().numpy() 
            result["labels"] = result["labels"][indices].detach().cpu().numpy() 
            result["boxes"] = result["boxes"][indices].detach().cpu().numpy() 

            prediction_string = ''

            for box, score, label in zip(result["boxes"], result["scores"], result["labels"]):
                prediction_string += str(label) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
                        box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '

            prediction_strings.append(prediction_string)
        
        return image_names, prediction_strings

# %%
valid[0]['pixel_values']

# %%



