#!/bin/bash

# Check if the correct number of arguments is provided
if [ "$#" -ne 1 ]; then
  echo "Usage: $0 <CKPT_DIR>"
  exit 1
fi
# Assign the first argument to CKPT_DIR
CKPT_DIR="$1"

bash scripts/omni_eval/4_vessel_det.sh $CKPT_DIR
bash scripts/omni_eval/7_NuInsSeg_organ_cls.sh $CKPT_DIR
bash scripts/omni_eval/10_CRC-MSI.sh $CKPT_DIR
bash scripts/omni_eval/13_liver_subtype.sh $CKPT_DIR