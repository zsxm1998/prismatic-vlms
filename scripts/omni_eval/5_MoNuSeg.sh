#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/5_MoNuSeg_nuclei_det.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/nucleus_segment/MoNuSeg/int_question_MoNuSeg_detect_04_test.jsonl"
else
  QUESTION_FILE="/c22073/datasets/nucleus_segment/MoNuSeg/question_MoNuSeg_detect_04_test.jsonl"
fi

ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/MoNuSeg_04_test.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/MoNuSeg_04_test"

if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/nucleus_segment/MoNuSeg/test/images_04 \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

python prismatic/eval/path_cv/no_class_detection.py \
    --img_dir /c22073/datasets/nucleus_segment/MoNuSeg/test/images_04 \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR > "$RES_FILE" 2>&1
