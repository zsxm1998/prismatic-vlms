#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/6_PanNuke_det.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

# ********************************PanNuke********************************
if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/nucleus_segment/PanNuke/detcls_int_question_PanNuke_part3.jsonl"
else
  QUESTION_FILE="/c22073/datasets/nucleus_segment/PanNuke/detcls_question_PanNuke_part3.jsonl"
fi

ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/detcls_PanNuke_part3.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/PanNuke_part3"

# Generate
if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/nucleus_segment/PanNuke/part3 \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

# Evaluate
echo "----------------------------------PanNuke----------------------------------" >> "$RES_FILE"
python prismatic/eval/path_cv/with_class_detection.py \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --img_dir /c22073/datasets/nucleus_segment/PanNuke/part3 \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1