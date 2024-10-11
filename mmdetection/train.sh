# 데이터셋 경로 설정
DATASET_ROOT='../../../dataset/'

# 로그 파일 및 모델 체크포인트 디렉토리 설정
WORK_DIR='mmdetection/work_dirs/vfnet_trash'
LOG_FILE="$WORK_DIR/training.log"

# 출력 디렉토리 생성
mkdir -p $WORK_DIR

# Python 스크립트 실행
python mmdetection/train.py --log-file=$LOG_FILE

# 학습 완료 후 메시지 출력
echo "Training completed! Logs can be found at $LOG_FILE"

# chmod +x mmdetection/train.sh