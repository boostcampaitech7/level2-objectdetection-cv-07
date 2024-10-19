import os
import json
from glob import glob

def get_id2label(classes):
    id2label = {id: label for id, label in enumerate(classes)}
    return id2label

def get_label2id(id2label):
    label2id = {label: id for id, label in id2label.items()}
    return label2id

def override_conf_if_exist(ref_conf, new_conf):
    for key, value in new_conf.items():
        if key in ref_conf:
            ref_conf[key] = value
    return ref_conf


def override_conf(ref_conf, new_conf):
    for key, value in new_conf.items():
        ref_conf[key] = value
    return ref_conf

def renew_if_path_exist(path):

    if not(os.path.exists(path)):
        return path
    else:
        for i in range(100000):
            temp_path = path + f'_{i}'
            if not(os.path.exists(temp_path)):
                return temp_path            
                
        raise Exception("path 지정을 실패했습니다.")


def save_log(trainer, log_path):    
    logs = ''
    for log in trainer.state.log_history:
        logs += json.dumps(log, indent=4) + "\n"

    with open(log_path, 'w') as f:
        f.write(logs)


def save_json(dicts, file_path):
    with open(file_path, "w") as f:
        json.dump(dicts, f, indent=4)


def read_json(file_path):
    with open(file_path, "r") as f:
        data = json.load(f)
    return data

def load_conf(default_conf_path, new_conf_path):
    default_conf = read_json(default_conf_path)
    new_conf = read_json(new_conf_path)
    conf = override_conf(default_conf, new_conf)
    return conf
    

def find_checkpoint_path(input_path):
    output_paths = glob(input_path + '/checkpoint*')

    if 1 < len(output_paths):
        raise Exception(f'checkpoint는 1개여야 합니다. {len(output_path)}개 있습니다.')

    output_path = output_paths[0]
    return output_path
