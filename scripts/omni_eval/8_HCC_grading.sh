#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/8_HCC_grading.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/HCC_grading/int_question_test.jsonl"
else
  QUESTION_FILE="/c22073/datasets/HCC_grading/question_test.jsonl"
fi

ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/HCC_grading.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/HCC_grading"

# Generate thumbnails
if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/HCC_grading/thumbnails \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

# Evaluate thumbnails
echo "----------------------------------thumbnails----------------------------------" > "$RES_FILE"
python ./prismatic/eval/path_cv/HCC_grading_eval.py \
    --img_dir /c22073/datasets/HCC_grading/thumbnails \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1


QUESTION_FILE="/c22073/datasets/HCC_grading/patch_question_test.jsonl"
ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/HCC_grading_patch.jsonl"

# Generate patch
if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/HCC_grading/patch_grading \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

# Evaluate patch
echo "----------------------------------patches----------------------------------" >> "$RES_FILE"
python ./prismatic/eval/path_cv/HCC_grading_eval.py \
    --img_dir /c22073/datasets/HCC_grading/patch_grading \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --patch >> "$RES_FILE" 2>&1