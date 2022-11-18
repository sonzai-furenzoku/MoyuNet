#!/usr/bin/env bash

TGT_LANG=$1
MODEL_DIR=$2
PROJ=$3

DATA_PATH="must-c/en-de"
W2V2_PATH="wav2vec_small.pt"
SPM_PATH="spm_unigram10000_st.model"

# download Wav2vec2 model

export CUDA_VISIBLE_DEVICES=0,1,2,3

# mkdir -p checkpoints
# wget -P checkpoints https://dl.fbaipublicfiles.com/fairseq/wav2vec/wav2vec_small.pt

mkdir -p ${MODEL_DIR}

fairseq-train $DATA_PATH \
    --user-dir MoyuST/MoyuST \
    --task speech_to_text_triplet_with_extra_mt \
    --train-subset train_st --valid-subset dev_st \
    --config-yaml config_st.yaml \
    --langpairs en-${TGT_LANG} --lang-prefix-tok \<lang:${TGT_LANG}\> \
    --max-audio-positions 600000 --max-source-positions 1024 --max-target-positions 1024 \
    --max-audio-tokens 1000000 --max-text-tokens 2000 --max-tokens 1000000  --max-tokens-valid 2000000 \
    --skip-invalid-size-inputs-valid-test \
    --text-data-sample-ratio 0.25 \
    \
    --arch moyunet --w2v2-model-path $W2V2_PATH \
    --using-attn \
    --optimizer adam --clip-norm 10.0 \
    --lr-scheduler inverse_sqrt --lr 1e-4  --warmup-updates 25000  --weight-decay 0.0 \
    \
    --criterion multi_task_cross_entropy_with_contrastive_with_extra_MT \
    --label-smoothing 0.1 --ignore-prefix-size 1 --report-accuracy \
    --contrastive-weight 0.0 --contrastive-temperature 0.02 --contrastive-seqlen-type none \
    \
    --keep-last-epochs 10 \
    --update-freq 2 --patience 5 \
    \
    --no-progress-bar --log-format json --log-interval 100 \
    --save-dir ${MODEL_DIR} \
    --ddp-backend=no_c10d --fp16 \
    \
    --eval-bleu --eval-bleu-args '{"beam": 4, "prefix_size": 1}' \
    --eval-bleu-detok moses --eval-bleu-remove-bpe \
    --eval-bleu-bpe sentencepiece --eval-bleu-bpe-path $SPM_PATH \
    --best-checkpoint-metric bleu --maximize-best-checkpoint-metric \
    --seed 3407
