import os
from pycocotools.coco import COCO
from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer
from functools import partial
import argparse
import utils
import dataset
import eval


def main(exp_path):
    current_path = os.path.dirname(os.path.abspath(__file__)) # main.py 폴더 경로
    default_conf_path = os.path.join(current_path, 'config/default.json') # 기본실험 설정파일 경로
    conf = utils.load_conf(default_conf_path, exp_path) # 기본설정(default.json)과 덮어쓰기 설정(exp_path) 불어오기

    # 실험결과 출력경로 지정
    conf['output_dir'] = os.path.join(current_path, 'result/' + conf['output_dir_format'].format(**conf)) # 결과 저장 경로
    conf['output_dir'] = utils.renew_if_path_exist(conf['output_dir'])

    # 로컬에서 불러올 모델이 지정되었으면, checkpoint_path 찾기
    if conf['saved_model_path'] is not None:
        conf['checkpoint_path'] = utils.find_checkpoint_path(conf['saved_model_path'])

    # conf 저장 (실험 설정)
    os.makedirs(conf['output_dir'], exist_ok=True)
    conf_path = os.path.join(conf['output_dir'], conf['output_dir'].split('/')[-1] + '.json')
    utils.save_json(conf, conf_path)

    # object detection 클래스 <-> 번호 변환 dictionary 생성
    id2label = utils.get_id2label(conf['classes'])
    label2id = utils.get_label2id(id2label)

    # 입력 이미지를 모델 입력에 맞게 변형
    # 입력 이미지 사이즈는 conf['image_size']로 결정
    image_processor = AutoImageProcessor.from_pretrained(
        conf['model_name'],
        do_resize=True,
        size={"max_height": conf['image_size'], "max_width": conf['image_size']},
        do_pad=True,
        pad_size={"height": conf['image_size'], "width": conf['image_size']},
    )

    # train/valid 불러오기
    train_info_path = os.path.join(conf['coco_dir_path'], conf['train_info_name'])
    valid_info_path = os.path.join(conf['coco_dir_path'], conf['valid_info_name'])
    
    coco_train = COCO(train_info_path)
    coco_valid = COCO(valid_info_path)
    train = dataset.COCO2dataset(conf['data_dir_path'], coco_train)
    valid = dataset.COCO2dataset(conf['data_dir_path'], coco_valid)

    # train/valid transform 정의
    train_augment_and_transform, validation_transform = dataset.get_transforms()

    # train/valid에 image_processor와 transform 정의
    train_transform_batch = partial(
        dataset.augment_and_transform_batch, transform=train_augment_and_transform, image_processor=image_processor
    )
    validation_transform_batch = partial(
        dataset.augment_and_transform_batch, transform=validation_transform, image_processor=image_processor
    )

    train = train.with_transform(train_transform_batch)
    valid = valid.with_transform(validation_transform_batch)
    
    # validation 결과 출력 메트릭 정의
    eval_compute_metrics_fn = partial(
        eval.compute_metrics, image_processor=image_processor, id2label=id2label, threshold=0.0
    )

    # 모델 불러오기 선택
    if conf['saved_model_path']: # 로컬에서 학습한 결과 불러오기
        model = AutoModelForObjectDetection.from_pretrained(
            conf['checkpoint_path'],
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )
    else: # hugging face에서 pretrained 모델 불러오기
        model = AutoModelForObjectDetection.from_pretrained(
            conf['model_name'],
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    # trainer 정의
    training_args = eval.load_train_args(conf)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=valid,
        tokenizer=image_processor,
        data_collator=dataset.collate_fn,
        compute_metrics=eval_compute_metrics_fn,
    )

    # train 시작
    trainer.train()

    # train 로그 저장
    log_path = os.path.join(conf['output_dir'], conf['output_dir'].split('/')[-1] + '.txt')
    utils.save_log(trainer, log_path)
    
    # test dataset 예측 및 결과 출력
    eval.test_eval(conf, model, image_processor) 

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