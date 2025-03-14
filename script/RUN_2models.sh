#!/bin/bash
# Run the evaluation script with arguments
python ../src/Model_swtich.py \
    --dataset_name "GSM8K" \
    --num_workers 250 \
    --Sampling True \
    --Sampling_Numbers 250\
    --results_sampling 5 \
    --modellist "gpt-4o-mini|gemini-1.5-flash-latest"\
    --ConsistencyThreshold 1  \
    --Open_SourceModel False \

