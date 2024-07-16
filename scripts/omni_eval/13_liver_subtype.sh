#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/13_liver_subtype.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

QUESTION_FILE="/c22073/datasets/ZheYi0607/liver_cancer/patch_question_test.jsonl"
ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/liver_subtype_patch_test.jsonl"

# ************************************* Evaluate for Patch *************************************
if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/ZheYi0607/liver_cancer/patches/ \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

echo -e "------------------------------Patch open ended classification------------------------------" > "$RES_FILE"
python prismatic/eval/path_cv/cls_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "liver_subtype" >> "$RES_FILE" 2>&1

echo -e "\n------------------------------Patch close ended classification------------------------------" >> "$RES_FILE"
python prismatic/eval/path_cv/choice_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "liver_subtype" >> "$RES_FILE" 2>&1

# ************************************* Evaluate for Thumbnails *************************************
if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/ZheYi0607/liver_cancer/int_thumbnail_question_test.jsonl"
else
  QUESTION_FILE="/c22073/datasets/ZheYi0607/liver_cancer/thumbnail_question_test.jsonl"
fi
ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/liver_subtype_thumbnail_test.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/liver_subtype_thumbnail"

# Generate thumbnails
if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/ZheYi0607/liver_cancer/thumbnails \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

# Evaluate thumbnails
echo "----------------------------------thumbnails----------------------------------" >> "$RES_FILE"
python ./prismatic/eval/path_cv/liver_subtype_eval.py \
    --img_dir /c22073/datasets/ZheYi0607/liver_cancer/thumbnails \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1
