import json
import pandas as pd

def json_to_grouped_pascal_voc_csv(json_path, output_csv_path):
    # Load the JSON file
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    # Dictionary to store Pascal VOC formatted rows by image_id
    voc_dict = {}
    
    for entry in data:
        image_id = f"test/{str(entry['image_id']).zfill(4)}.jpg"
        bbox = entry['bbox']
        score = entry['score']
        category_id = entry['category_id']
        
        # Convert COCO bbox format [xmin, ymin, width, height] to [xmin, ymin, xmax, ymax]
        xmin = bbox[0]
        ymin = bbox[1]
        xmax = bbox[0] + bbox[2]
        ymax = bbox[1] + bbox[3]
        
        # Format as: class_id score xmin ymin xmax ymax
        prediction_string = f"{category_id} {score} {xmin} {ymin} {xmax} {ymax}"
        
        # If the image_id is already in the dict, append the new prediction string
        if image_id in voc_dict:
            voc_dict[image_id] += " " + prediction_string
        else:
            voc_dict[image_id] = prediction_string
    
    # Create a list of rows for DataFrame
    voc_rows = [{"PredictionString": pred_str, "image_id": img_id} for img_id, pred_str in voc_dict.items()]
    
    # Create DataFrame
    voc_df = pd.DataFrame(voc_rows)
    
    # Save as CSV
    voc_df.to_csv(output_csv_path, index=False)
    print(f"CSV saved at {output_csv_path}")

# Example usage
json_path = '/data/ephemeral/home/Jeongseon/mmdetection/V3/work_dirs/coco_detection/test.bbox.json'  # Replace with your actual path
output_csv_path = '/data/ephemeral/home/Jeongseon/mmdetection/V3/work_dirs/pascal_voc_format_7.csv'  # Replace with your desired output path
json_to_grouped_pascal_voc_csv(json_path, output_csv_path)  # Uncomment and update the paths to run this
