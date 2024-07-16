#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <CKPT_DIR>"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"
CKPT_NAME=$(basename "$CKPT_DIR")
RES_FILE="./playground/omnipath/eval/$CKPT_NAME/0_language_benchmark.log"

mkdir -p ./playground/omnipath/eval/$CKPT_NAME


# PVQA
python prismatic/eval/model_vqa.py \
    --model_path $CKPT_DIR \
    --question_file /c22073/datasets/PathVQA/pvqa_test_wo_ans.jsonl \
    --image_folder /c22073/datasets/PathVQA/images/test \
    --answers_file ./playground/omnipath/eval/$CKPT_NAME/PathVQA.jsonl \
    --num_chunks 1 \
    --chunk_idx 0 \
    --do_sample False \
    --temperature 0

# Evaluate
# PVQA
echo "------------------------------PathVQA------------------------------" > "$RES_FILE"
python ./prismatic/eval/path_benchmark/quilt_eval.py \
    --gt /c22073/datasets/PathVQA/pvqa_test_w_ans.json \
    --pred ./playground/omnipath/eval/$CKPT_NAME/PathVQA.jsonl >> "$RES_FILE" 2>&1


# PMCVQA
# python -m llava.eval.model_vqa_science \
#     --model_path $CKPT_DIR \
#     --question_file /c22073/datasets/PMC-VQA/pmcvqa_test_wo_ans.json \
#     --image_folder /c22073/datasets/PMC-VQA/images \
#     --answers_file ./playground/omnipath/eval/$CKPT_NAME/PMC-VQA.jsonl \
#     --single-pred-prompt \
#     --do_sample False \
#     --temperature 0

# Evaluate
# PMCVQA
# echo -e "\n------------------------------PMC-VQA------------------------------" >> "$RES_FILE"
# python ./prismatic/eval/path_benchmark/pmc_eval.py \
#     --question_file /c22073/datasets/PMC-VQA/pmcvqa_test_wo_ans.json \
#     --result-file ./playground/omnipath/eval/$CKPT_NAME/PMC-VQA.jsonl \
#     --output-file ./playground/omnipath/eval/$CKPT_NAME/PMC-VQA_output.jsonl \
#     --output-result ./playground/omnipath/eval/$CKPT_NAME/PMC-VQA_result.json >> "$RES_FILE" 2>&1


# QUILT-VQA
python prismatic/eval/model_vqa.py \
    --model_path $CKPT_DIR \
    --question_file /c22073/datasets/QuiltVQA/quiltvqa_test_wo_ans.jsonl \
    --image_folder /c22073/datasets/QuiltVQA/images \
    --answers_file ./playground/omnipath/eval/$CKPT_NAME/QuiltVQA.jsonl \
    --num_chunks 1 \
    --chunk_idx 0 \
    --do_sample False \
    --temperature 0

# Evaluate
# QUILT-VQA
echo -e "\n------------------------------QUILT-VQA------------------------------" >> "$RES_FILE"
python ./prismatic/eval/path_benchmark/quilt_eval.py \
    --quilt True \
    --gt /c22073/datasets/QuiltVQA/quiltvqa_test_w_ans.json \
    --pred ./playground/omnipath/eval/$CKPT_NAME/QuiltVQA.jsonl >> "$RES_FILE" 2>&1


# QUILT-VQA RED
python prismatic/eval/model_vqa.py \
    --model_path $CKPT_DIR \
    --question_file /c22073/datasets/QuiltVQA/quiltvqa_red_test_wo_ans.jsonl \
    --image_folder /c22073/datasets/QuiltVQA/red_circle \
    --answers_file ./playground/omnipath/eval/$CKPT_NAME/QuiltVQA-red-circle.jsonl \
    --num_chunks 1 \
    --chunk_idx 0 \
    --do_sample False \
    --temperature 0

# QUILT-VQA No RED
python prismatic/eval/model_vqa.py \
    --model_path $CKPT_DIR \
    --question_file /c22073/datasets/QuiltVQA/quiltvqa_nored_test_wo_ans.jsonl \
    --image_folder /c22073/datasets/QuiltVQA/images \
    --answers_file ./playground/omnipath/eval/$CKPT_NAME/QuiltVQA-nored-circle.jsonl \
    --num_chunks 1 \
    --chunk_idx 0 \
    --do_sample False \
    --temperature 0


# Evaluate
# QUILT-VQA  RED
echo -e "\n------------------------------Quilt-VQA Red Circle------------------------------" >> "$RES_FILE"
python ./prismatic/eval/path_benchmark/quilt_eval.py \
    --quilt True \
    --gt /c22073/datasets/QuiltVQA/quiltvqa_red_test_w_ans.json \
    --pred ./playground/omnipath/eval/$CKPT_NAME/QuiltVQA-red-circle.jsonl >> "$RES_FILE" 2>&1

# QUILT-VQA No RED
echo -e "\n------------------------------Quilt-VQA No Red Circle------------------------------" >> "$RES_FILE"
python ./prismatic/eval/path_benchmark/quilt_eval.py \
    --quilt True \
    --gt /c22073/datasets/QuiltVQA/quiltvqa_red_test_w_ans.json \
    --pred ./playground/omnipath/eval/$CKPT_NAME/QuiltVQA-nored-circle.jsonl >> "$RES_FILE" 2>&1
