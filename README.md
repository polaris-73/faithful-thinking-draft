# Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models

This repository contains evaluation code for "Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models".

## Dataset

The benchmarking data can be found from the Hugging Face Hub: 
- `https://huggingface.co/datasets/polaris-73/faithful-thinking-draft`

## Installation

```bash
git clone https://github.com/polaris-73/faithful-thinking-draft.git
cd faithful-thinking-draft
pip install -r requirements.txt
```

## Reproducing experiments
```bash
bash scripts/run_eval.sh
bash scripts/agg_results.sh
```

## An Example of Detailed Evaluation

### Evaluation Running
```bash
python faithful_thinking_draft/eval/eval.py \
  --api_type vllm \
  --model_name [MODEL_NAME] \
  --dataset_type gpqa \     
  --output_dir ./results/ \
  --temperature 0 \
  --benchmark_trace deepseek_reasoner \ 
  --intervention_types all
```

### Post-processing results

```bash
python faithful_thinking_draft/eval/eval.py \
  --api_type vllm \
  --model_name [MODEL_NAME] \
  --dataset_type gpqa \     
  --output_dir ./results/ \
  --temperature 0 \
  --benchmark_trace deepseek_reasoner \ 
  --intervention_types all \
  --eval_only \
  --resume \
  --use_llm_judge \
  --judge_model_name "Qwen/Qwen2.5-32B-Instruct"
```

### Aggregating Results

```bash
python faithful_thinking_draft/eval/agg_score.py \
  --evaluation_models [MODEL_NAME] \
  --results_dir ./results/ \
  --dataset_types gpqa \
  --benchmark_traces deepseek_reasoner
```

## Key Parameters

- `--api_type`: Model API to use (vllm, deepseek)
- `--model_name`: Model to evaluate
- `--dataset_type`: Dataset for evaluation (gpqa, mmlu)
- `--intervention_types`: Types of interventions to evaluate (Intra-Draft Faithfulness: "original", "corrupt_option", "shift_mapping", Draft-to-Answer Faithfulness: "direct_alternative_conclusion", "plausible_alternative_conclusion")
- `--benchmark_trace`: Benchmark trace to evaluate on (deepseek_reasoner, Qwen3_32B)
- `--step_locations`: Step locations for Intra-Draft Faithfulness (first_step, middle_step, end_step)
- `--step_types`: Step types for Intra-Draft Faithfulness (backtrack, continue)
- `--answer_types`: Answer types for Draft-to-Answer Faithfulness (standard, immediate)

## Citation
```
@article{xiong2025measuring,
  title={Measuring the Faithfulness of Thinking Drafts in Large Reasoning Models},
  author={Xiong, Zidi and Chen, Shan and Qi, Zhenting and Lakkaraju, Himabindu},
  journal={arXiv preprint arXiv:2505.13774},
  year={2025}
}
``` 