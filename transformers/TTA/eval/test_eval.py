from tqdm import tqdm
import pandas as pd
import numpy as np
import torch

def test_eval(model, image_processor, coco_data, data, tta, result_path, batch_size=1):
    image_names = []
    prediction_strings = []
    batch_indices = get_batch_indices(batch_size, len(data))

    for batch_index in tqdm(batch_indices):
        batch = data[batch_index]
        image_name, prediction_string = get_predictions(model, image_processor, coco_data, batch, tta)
        
        image_names.extend(image_name)
        prediction_strings.extend(prediction_string)

    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = image_names
    submission.to_csv(result_path, index=None)

        

def get_predictions(model, image_processor, coco_data, batch, tta, threshold=0.05, device="cuda"):
    with torch.no_grad():
        images = [np.array(image) for image in batch['image']]
        images = tta(images=images) # image TTA 적용
        image_ids = batch['image_id']
        image_infos = coco_data.loadImgs(image_ids)

        inputs = image_processor(images=images, return_tensors="pt")
        outputs = model(**inputs.to(device))
        target_sizes = [image.shape for image in images]
        results = image_processor.post_process_object_detection(outputs, threshold=threshold, target_sizes=target_sizes)
        results = tta(results=results, target_sizes=target_sizes) # results에 있는 boxes 원래 좌표로 변환

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
    

def get_batch_indices(batch_size, n):
    i = 0
    batch_index = []
    batch_indices = []

    while True:
        if (i == n): 
            if len(batch_index):
                batch_indices.append(batch_index)
            break

        batch_index.append(i)

        if (len(batch_index) % batch_size == 0):
            batch_indices.append(batch_index)
            batch_index = []
            
        i += 1  

    return batch_indices
