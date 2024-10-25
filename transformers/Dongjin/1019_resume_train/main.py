import os
from pycocotools.coco import COCO
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer
from functools import partial
import argparse
import utils
import dataset
import train_eval
import torch

def main(exp_path):
    current_path = os.path.dirname(os.path.abspath(__file__))
    default_conf_path = os.path.join(current_path, 'config/default.json')
    conf = utils.load_conf(default_conf_path, exp_path)

    conf['output_dir'] = os.path.join(current_path, 'result/' + conf['output_dir_format'].format(**conf)) # 결과 저장 경로
    conf['output_dir'] = utils.renew_if_path_exist(conf['output_dir'])

    if conf['saved_model_path']:
        conf['checkpoint_path'] = utils.find_checkpoint_path(conf['saved_model_path'])

    # conf 저장
    os.makedirs(conf['output_dir'], exist_ok=True)
    conf_path = os.path.join(conf['output_dir'], conf['output_dir'].split('/')[-1] + '.json')
    utils.save_json(conf, conf_path)

    train_info_path = os.path.join(conf['coco_dir_path'], conf['train_info_name'])
    valid_info_path = os.path.join(conf['coco_dir_path'], conf['valid_info_name'])

    coco_train = COCO(train_info_path)
    coco_valid = COCO(valid_info_path)

    id2label = utils.get_id2label(conf['classes'])
    label2id = utils.get_label2id(id2label)

    train = dataset.COCO2dataset(conf['data_dir_path'], coco_train, range(10))
    valid = dataset.COCO2dataset(conf['data_dir_path'], coco_valid, range(10))

    train_augment_and_transform, validation_transform = dataset.get_transforms()

    image_processor = AutoImageProcessor.from_pretrained(
        conf['model_name'],
        do_resize=True,
        size={"max_height": conf['image_size'], "max_width": conf['image_size']},
        do_pad=True,
        pad_size={"height": conf['image_size'], "width": conf['image_size']},
    )

    # Make transform functions for batch and apply for dataset splits
    train_transform_batch = partial(
        dataset.augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        dataset.augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    train = train.with_transform(train_transform_batch)
    valid = valid.with_transform(validation_transform_batch)

    eval_compute_metrics_fn = partial(
        train_eval.compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
    )

    if conf['saved_model_path']:
        model = AutoModelForObjectDetection.from_pretrained(
            conf['checkpoint_path'],
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    else:
        model = AutoModelForObjectDetection.from_pretrained(
            conf['model_name'],
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    training_args = train_eval.load_train_args(conf)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=image_processor,
        data_collator=dataset.collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    trainer.train()

    log_path = os.path.join(conf['output_dir'], conf['output_dir'].split('/')[-1] + '.txt')
    utils.save_log(trainer, log_path)
    
    train_eval.test_eval(conf, model, image_processor) 


if __name__ == "__main__":
    # Transformers object detection training script
    # 실험 설정은 config/default.json를 이용하여 동작합니다.
    # --exp_path를 이용하여 default.json의 설정을 덮어쓰기할 수 있습니다.
    # default.json으로 기본 설정을 하고, 세부 실험 조건을 설정하는 json 파일을 이용하여
    # 여러 실험을 쉽게 돌릴 수 있습니다. (예시: config/exp1.json)

    parser = argparse.ArgumentParser(description='Transformers training script')
    parser.add_argument('--exp_path', type=str, default=None, help='exp.json file path (optional)')
    args = parser.parse_args()
    exp_path = args.exp_path

    main(exp_path)