#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/2_mvi_set2_cvpr_test.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file /c22073/datasets/VLM_MVI/set2/mvi_cvpr/question.jsonl \
      --image_folder /c22073/datasets/VLM_MVI/set2/mvi_cvpr/test \
      --answers_file ./playground/omnipath/eval/$CKPT_NAME/mvi_set2_cvpr_test.jsonl \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

python prismatic/eval/path_cv/mvi_cvpr_eval.py \
    --gt /c22073/datasets/VLM_MVI/set2/mvi_cvpr/question.jsonl \
    --pred ./playground/omnipath/eval/$CKPT_NAME/mvi_set2_cvpr_test.jsonl > "$RES_FILE" 2>&1
