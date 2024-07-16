#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/14_LN_zheyi.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

# ************************************* Evaluate for lymph node detection *************************************
if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/ZheYi0607/LN_det/int_question_test.jsonl"
else
  QUESTION_FILE="/c22073/datasets/ZheYi0607/LN_det/question_test.jsonl"
fi
ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/LN_det.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/LN_det"

if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/ZheYi0607/LN_det/images \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

echo "----------------------------------LN_det----------------------------------" > "$RES_FILE"
python prismatic/eval/path_cv/no_class_detection.py \
    --img_dir /c22073/datasets/ZheYi0607/LN_det/images \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1

# ************************************* Evaluate for lymph node metastasis cls and seg *************************************
if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/ZheYi0607/LNM/int_question_test.jsonl"
else
  QUESTION_FILE="/c22073/datasets/ZheYi0607/LNM/question_test.jsonl"
fi
ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/LNM.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/LNM"

if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/ZheYi0607/LNM/images \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

echo "----------------------------------LNM classification----------------------------------" >> "$RES_FILE"
python prismatic/eval/path_cv/cls_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "LNM" >> "$RES_FILE" 2>&1

echo "----------------------------------LNM segmentation----------------------------------" >> "$RES_FILE"
python prismatic/eval/path_cv/no_class_segmentation.py \
    --img_dir /c22073/datasets/ZheYi0607/LNM/images \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1