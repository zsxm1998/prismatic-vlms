#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/3_mvi_nuclei.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/VLM_MVI/set1/mvi_cancerous_nucleus/int_question_test_convs.jsonl"
else
  QUESTION_FILE="/c22073/datasets/VLM_MVI/set1/mvi_cancerous_nucleus/question_test_convs.jsonl"
fi

ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/mvi_set1_cancerous_nucleus.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/mvi_set1_cancerous_nucleus"

if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/VLM_MVI/set1/mvi_cancerous_nucleus/images \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

python prismatic/eval/path_cv/no_class_detection.py \
    --img_dir /c22073/datasets/VLM_MVI/set1/mvi_cancerous_nucleus/images \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR > "$RES_FILE" 2>&1
