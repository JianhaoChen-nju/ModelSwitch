#!/bin/bash
# Run the evaluation script with arguments
python ../src/Model_swtich.py \
    --dataset_name "GSM8K" \
    --num_workers 250 \
    --Sampling True \
    --Sampling_Numbers 250\
    --results_sampling 5 \
    --modellist "Llama-3.1-8B-Instruct|gemma-2-9b-it"\
    --ConsistencyThreshold 1  \
    --Open_SourceModel True \

