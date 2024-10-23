from tqdm import tqdm
import pandas as pd
import os
import dataset
from pycocotools.coco import COCO
import torch

def test_eval(conf, model, image_processor):
    batch_size = 8
    test_info_name = 'test.json'

    test_info_path = os.path.join(conf['data_dir_path'], test_info_name)
    result_path = os.path.join(conf['output_dir'], conf['output_dir'].split('/')[-1] + '.csv')

    coco_test = COCO(test_info_path)
    test = dataset.COCO2dataset(conf['data_dir_path'], coco_test)

    image_names = []
    prediction_strings = []
    batch_indices = get_batch_indices(batch_size, len(test))

    for batch_index in tqdm(batch_indices):
        batch = test[batch_index]
        image_name, prediction_string = get_predictions(batch, coco_test, model, image_processor)
        
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
        

def get_predictions(batch, coco_test, model, image_processor):
    threshold = 0.05
    device = "cuda"

    with torch.no_grad():
        images = batch['image']
        image_ids = batch['image_id']
        image_infos = coco_test.loadImgs(image_ids)

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