import os
from pycocotools.coco import COCO
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer
from functools import partial
import argparse
import utils
import dataset
import train_eval

def main(exp_conf_path):
    current_path = os.path.dirname(os.path.abspath(__file__))
    default_conf_path = os.path.join(current_path, 'config/default.json')
    conf = utils.load_conf(default_conf_path, exp_conf_path)

    conf['output_dir'] = os.path.join(current_path, 'result/' + conf['output_dir_format'].format(**conf)) # 결과 저장 경로
    conf['output_dir'] = utils.renew_if_path_exist(conf['output_dir'])

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

    train = dataset.COCO2dataset(conf['data_dir_path'], coco_train)
    valid = dataset.COCO2dataset(conf['data_dir_path'], coco_valid)

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
    parser = argparse.ArgumentParser(description='TIMM training script')
    parser.add_argument('exp_conf_path', type=str, help='exp.json file path')
    args = parser.parse_args()
    exp_conf_path = args.exp_conf_path

    main(exp_conf_path)