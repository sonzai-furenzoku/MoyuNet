#!/bin/bash
set -eo pipefail

DATA_PATH="must-c/en-de"
CKPT_PATH="averaging.pt"

export CUDA_VISIBLE_DEVICES=0

fairseq-generate $DATA_PATH \
    --user-dir MoyuST/MoyuST \
    --gen-subset tst-COMMON_st --task speech_to_text --prefix-size 1 \
    --quiet \
    --lenpen 0.7 \
    --batch-size 32 --max-source-positions 4000000 --beam 10 \
    --config-yaml config_st.yaml  --path $CKPT_PATH \
    --scoring sacrebleu
