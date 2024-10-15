MODEL="yolox_s_8x8_300e_coco"

python inference.py \
    --model $MODEL \
    --root '/data/ephemeral/home/data/dataset/' \
    --num_classes 10 \

# chmod +x ./inference.sh
# ./inference.sh