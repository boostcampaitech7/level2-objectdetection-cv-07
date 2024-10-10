# data_utils.py
import json
from sklearn.model_selection import train_test_split

def load_annotations(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

def split_data(data, test_size=0.2, random_state=42):
    train_data, val_data = train_test_split(data['images'], test_size=test_size, random_state=random_state)
    return train_data, val_data

def save_data(data, filepath):
    with open(filepath, 'w') as f:
        json.dump(data, f)

def prepare_datasets(root, data, train_data, val_data):
    # train dataset
    data['images'] = train_data
    save_data(data, f'{root}/train_split.json')
    
    # validation dataset
    data['images'] = val_data
    save_data(data, f'{root}/val_split.json')