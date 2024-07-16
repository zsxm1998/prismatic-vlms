#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -lt 1 ]; then
  echo "Usage: $0 <CKPT_DIR> [-ng]"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/15_NI_zheyi.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME

# ************************************* Evaluate for nerve detection *************************************
if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/ZheYi0607/NI_det/int_question_test.jsonl"
else
  QUESTION_FILE="/c22073/datasets/ZheYi0607/NI_det/question_test.jsonl"
fi
ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/NI_det.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/NI_det"

if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/ZheYi0607/NI_det/images \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

echo "----------------------------------NI_det----------------------------------" > "$RES_FILE"
python prismatic/eval/path_cv/no_class_detection.py \
    --img_dir /c22073/datasets/ZheYi0607/NI_det/images \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1

# ************************************* Evaluate for neural invasion *************************************
if [[ $CKPT_DIR == *"int"* ]]; then
  QUESTION_FILE="/c22073/datasets/ZheYi0607/NI_cls/int_question_test.jsonl"
else
  QUESTION_FILE="/c22073/datasets/ZheYi0607/NI_cls/question_test.jsonl"
fi
ANSWER_FILE="./playground/omnipath/eval/$CKPT_NAME/NI_cls.jsonl"
VIS_DIR="output/omnipath/$CKPT_NAME/NI_cls"

if [[ " $@ " != *" -ng "* ]]; then
  python prismatic/eval/model_vqa.py \
      --model_path $CKPT_DIR \
      --question_file $QUESTION_FILE \
      --image_folder /c22073/datasets/ZheYi0607/NI_cls/images \
      --answers_file $ANSWER_FILE \
      --num_chunks 1 \
      --chunk_idx 0 \
      --do_sample False \
      --temperature 0
fi

echo "----------------------------------neural invasion classification----------------------------------" >> "$RES_FILE"
python prismatic/eval/path_cv/cls_eval.py \
    --gt $QUESTION_FILE \
    --pred $ANSWER_FILE \
    --dataset "NI" >> "$RES_FILE" 2>&1

echo "----------------------------------neural invasion segmentation----------------------------------" >> "$RES_FILE"
python prismatic/eval/path_cv/no_class_segmentation.py \
    --img_dir /c22073/datasets/ZheYi0607/NI_cls/images \
    --q_file $QUESTION_FILE \
    --a_file $ANSWER_FILE \
    --vis_dir $VIS_DIR >> "$RES_FILE" 2>&1