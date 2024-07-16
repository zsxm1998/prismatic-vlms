#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/6_NuCLS_det.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

# ********************************NuCLS********************************
if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/nucleus_segment/NuCLS/detcls_int_question_NuCLS_test.jsonl"
else
  QUESTION_FILE="/c22073/datasets/nucleus_segment/NuCLS/detcls_question_NuCLS_test.jsonl"
fi

ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/detcls_NuCLS_test.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/NuCLS_test"

# Generate
if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/nucleus_segment/NuCLS/images \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

# Evaluate
echo "----------------------------------NuCLS----------------------------------" > "$RES_FILE"
python prismatic/eval/path_cv/with_class_detection.py \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --img_dir /c22073/datasets/nucleus_segment/NuCLS/images \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1
