#!/bin/bash

datasets=("Apache" "BGL" "HDFS" "HPC" "Hadoop" "HealthApp" "Linux" "Mac" "OpenSSH" "OpenStack" "Proxifier" "Spark" "Thunderbird" "Zookeeper")

shot=32
model="./models/roberta-base"

for iteration in {1..5}; do
    echo "Starting iteration $iteration..."

    mkdir -p ../results_$iteration
    mkdir -p ../results_$iteration/logs
    mkdir -p ../results_$iteration/samples

    for dataset in "${datasets[@]}"; do
        echo "Processing dataset: $dataset"

        python3 log_parser.py \
            --model_name_or_path $model \
            --log_file ../datasets/loghub-2.0-full/$dataset/${dataset}_full.log_structured.csv \
            --train_file ../datasets/loghub-2.0-full/$dataset/sampled_sim_${shot}.json \
            --shot ${shot} \
            --dataset_name $dataset \
            --parsing_num_processes 1 \
            --output_dir ../results_$iteration \
            --max_train_steps 1000 \
            --load_model False \
            --save_model False

        python3 evaluation.py \
            --output_dir ../results_$iteration \
            --dataset "$dataset"
        
        cp ../datasets/loghub-2.0-full/$dataset/sampled_sim_${shot}.json ../results_$iteration/samples/${dataset}_sampled_sim_${shot}.json
        echo "Copied ../datasets/loghub-2.0-full/$dataset/sampled_sim_${shot}.json to ../results_$iteration/samples/${dataset}_sampled_sim_${shot}.json"
    done

done

echo "All iterations completed."