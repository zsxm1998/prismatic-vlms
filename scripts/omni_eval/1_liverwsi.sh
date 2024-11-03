#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/1_liverWSI.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/liverWSI/origin_patch_bbox_contour_336_14/int_question_edge01_val.jsonl"
else
  QUESTION_FILE="/c22073/datasets/liverWSI/origin_patch_bbox_contour_336_14/question_edge01_val.jsonl"
fi

ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/liverWSI_edge01.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/liverWSI_edge01"

# Generate
if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --image_folder /c22073/datasets/liverWSI/edge01/images \
      --question_file $QUESTION_FILE \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

# Evaluate
python ./prismatic/eval/path_cv/liverWSI_eval.py \
    --img_dir_format /c22073/datasets/liverWSI/{c}/images \
    --q_file_format "${QUESTION_FILE/edge01/\{c\}}" \
    --a_file_format "${ANSWER_FILE/edge01/\{c\}}" \
    --vis_dir_format "${VIS_DIR/edge01/\{c\}}" > "$RES_FILE" 2>&1
