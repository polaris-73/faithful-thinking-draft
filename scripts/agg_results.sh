EVAL_MODELS=("DeepSeek-R1-Distill-Qwen-7B" "DeepSeek-R1-Distill-Llama-8B" "DeepSeek-R1-Distill-Qwen-14B" "DeepSeek-R1-Distill-Qwen-32B" "QwQ-32B" "Skywork-OR1-32B-Preview")

python eval/agg_score.py --results_dir ./results/ --evaluation_models ${EVAL_MODELS[@]}




