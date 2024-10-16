import numpy as np
import csv

def wbf(boxes, scores, iou_threshold=0.5):
    """
    Apply Weighted Box Fusion to combine bounding boxes.
    
    Parameters:
        boxes (list): List of bounding boxes in the format [[x1, y1, x2, y2], ...]
        scores (list): List of scores corresponding to each box
        iou_threshold (float): IoU threshold for merging boxes
        
    Returns:
        List of combined boxes in the same format.
    """
    if len(boxes) == 0:
        return []
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort boxes by score
    sorted_indices = np.argsort(scores)[::-1]
    boxes = boxes[sorted_indices]
    
    selected_boxes = []
    selected_scores = []

    while len(boxes) > 0:
        # Select the box with the highest score
        current_box = boxes[0]
        selected_boxes.append(current_box)
        selected_scores.append(scores[sorted_indices[0]])

        # Calculate IoU of the current box with the remaining boxes
        x1 = np.maximum(current_box[0], boxes[:, 0])
        y1 = np.maximum(current_box[1], boxes[:, 1])
        x2 = np.minimum(current_box[2], boxes[:, 2])
        y2 = np.minimum(current_box[3], boxes[:, 3])

        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        
        iou = intersection / (area_current + area_boxes - intersection)

        # Select boxes with IoU less than the threshold
        keep = iou < iou_threshold

        # Update boxes and scores
        boxes = boxes[keep]
        scores = scores[keep]
        sorted_indices = sorted_indices[keep]

    # Combine selected boxes using weighted average
    if len(selected_boxes) > 0:
        combined_box = np.mean(np.array(selected_boxes), axis=0).tolist()
        return [combined_box]

    return []

def save_to_csv(combined_boxes, filename='combined_boxes.csv'):
    """
    Save combined boxes to a CSV file.
    
    Parameters:
        combined_boxes (list): List of combined boxes to save.
        filename (str): Name of the output CSV file.
    """
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['x1', 'y1', 'x2', 'y2'])  # Header
        for box in combined_boxes:
            writer.writerow(box)

# Example usage
boxes = [
    [50, 50, 150, 150],
    [60, 60, 160, 160],
    [70, 70, 170, 170],
    [200, 200, 300, 300]
]

scores = [0.9, 0.8, 0.75, 0.95]

combined_boxes = wbf(boxes, scores)
print("Combined Boxes:", combined_boxes)

# Save the combined boxes to a CSV file
save_to_csv(combined_boxes)