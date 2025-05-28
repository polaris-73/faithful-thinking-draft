#!/bin/bash
DATASET=("mmlu" "gpqa")
MODELNAME=("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B" "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "Qwen/QwQ-32B" "Skywork/Skywork-OR1-32B-Preview")
BENCHMARK_TRACE=("Qwen3_32B" "deepseek_reasoner")

API_TYPE="vllm"

for mn in ${MODELNAME[@]}; do
    for dt in ${DATASET[@]}; do
        for bt in ${BENCHMARK_TRACE[@]}; do
    
            python eval/eval.py \
                --model_name $mn \
                --api_type $API_TYPE \
                --temperature 0.0 \
                --dataset_type $dt \
                --intervention_types all \
                --benchmark_trace $bt \

            python eval/eval.py \
                --model_name $mn \
                --api_type $API_TYPE \
                --temperature 0.0 \
                --dataset_type $dt \
                --intervention_types all \
                --benchmark_trace $bt \
                --eval_only \
                --resume \
                --use_llm_judge \
                --judge_model_name "Qwen/Qwen2.5-32B-Instruct"
        done    
    done
done

