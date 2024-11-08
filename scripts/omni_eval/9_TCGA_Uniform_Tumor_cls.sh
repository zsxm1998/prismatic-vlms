#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/9_TCGA_Uniform_Tumor.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

QUESTION_FILE="/c22073/datasets/TCGA_Uniform_Tumor_subset/question_test.jsonl"
ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/TCGA_Uniform_Tumor_test.jsonl"

if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/TCGA_Uniform_Tumor_subset/ \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

python prismatic/eval/path_cv/cls_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "TCGA_Uniform_Tumor" > "$RES_FILE" 2>&1
