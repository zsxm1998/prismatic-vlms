#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <CKPT_DIR>"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"

bash scripts/omni_eval/14_LN_zheyi.sh $CKPT_DIR
bash scripts/omni_eval/0_language_all.sh $CKPT_DIR