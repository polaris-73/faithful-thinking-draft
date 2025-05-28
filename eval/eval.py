import os
import json
import torch
import logging
from tqdm import tqdm
import argparse
import re
import time
from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM


# Import the modular components
from generation_call import (
    call_deepseek_api, 
    call_vllm_batch,
)
from llm_judge_utils import extract_answer_llm, judge_behaviors
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')



def extract_answer(text, answer_only=False):
    """Extract answer from GPQA response"""
    # Extract thinking and answer parts
    if "</think>" in text:
        raw_thinking = text.split("</think>")[0]
        raw_answer = text.split("</think>")[1]
    else:
        raw_thinking = text
        raw_answer = ""
    if answer_only and "</think>" not in text:
        raw_answer = text
    
    # Try to extract the answer letter
    pattern = r'\*\*([A-D])\*\*'
    match = re.search(pattern, raw_answer)
    if match:
        return match.group(1).upper(), raw_thinking, raw_answer
    
    # Fallback patterns if tag is not found
    patterns = [
        r'Answer:\s*([A-D])',
        r'([A-D])\)',
        r'\b([A-D])\b',
        r'The answer is: ([A-D])',
    ]
    
    for pattern in patterns:
        match = re.search(pattern, raw_answer)
        if match:
            return match.group(1).upper(), raw_thinking, raw_answer
    
    return None, raw_thinking, raw_answer

def load_dataset_by_type(dataset_type, benchmark_trace):
    """Load dataset based on type"""
    if dataset_type in ["gpqa", "mmlu"] and benchmark_trace in ["deepseek_reasoner", "Qwen3_32B"]:
        dataset = load_dataset("polaris-73/faithful-thinking-draft", f"{benchmark_trace}_{dataset_type}")['test']
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type} and benchmark trace: {benchmark_trace}")
    
    return dataset

def call_model_batch(api_type, prompts, model_name, thinking_list=None, model=None, tokenizer=None, vllm_model=None, temperature=0, answer_type=None, thinking_end=False):
    """Call the appropriate model with batch processing and multiple outputs per prompt"""
    if answer_type == "immediate":
        immediate_answer = True
    else:
        immediate_answer = False
    if api_type == "vllm":
        # VLLM supports native batching with multiple outputs
        return call_vllm_batch(prompts, model_name, vllm_model, tokenizer, thinking_list, temperature, thinking_end=thinking_end, immediate_answer=immediate_answer)
    
    else:
        # Fallback to individual processing for API calls
        all_responses = []
        for i, prompt in enumerate(prompts):
            thinking = thinking_list[i] if thinking_list and i < len(thinking_list) else None
            try:
                if api_type == "deepseek":
                    responses = call_deepseek_api(prompt, model_name, thinking, temperature, thinking_end=thinking_end, immediate_answer=immediate_answer)
                else:
                    raise ValueError(f"Unsupported API type: {api_type}")
                
                all_responses.append(responses)
                
                # Add delay for API calls to respect rate limits
                if api_type in ["deepseek"]:
                    time.sleep(1)
                    
            except Exception as e:
                logging.error(f"Error in API call for prompt {i}: {str(e)}")
                all_responses.append([None])
        
        return all_responses

def evaluate(evaluation_data, api_type, model_name, model=None, tokenizer=None, vllm_model=None, temperature=0, answer_type=None, thinking_end=False, dry_run=False):
    """Evaluate faithfulness for a batch of questions with multiple outputs per prompt
    
    Args:
        evaluation_data: List of dictionaries with data for each question
        api_type: Type of API to use (vllm, deepseek)
        model_name: Name of the model to use
        model: Model instance if available
        tokenizer: Tokenizer instance
        vllm_model: VLLM model instance if available
        temperature: Temperature for generation
        intervention_key: Type of intervention applied
        answer_type: Type of answer generation (standard, immediate)
        thinking_end: Whether to end the thinking generation
        dry_run: Whether to use mock responses for testing
        
    Returns:
        List of result dictionaries
    """
    # Prepare prompts and thinking for each question
    prompts = []
    thinking_list = []
    correct_answers = []
    
    for i, data in enumerate(evaluation_data):
        formatted_prompt = data["formatted_question"]
        formatted_thinking = data["perturbed_thinking"]
        correct_answer = data["correct_answer"]
        # Format the example
        prompts.append(formatted_prompt)
        thinking_list.append(formatted_thinking)
        correct_answers.append(correct_answer)
    # Call model in batch with multiple outputs per prompt
    if dry_run:
        batch_responses = [["Hmm, I think the answer is A </think>The answer is **A**"]] * len(prompts)
    else:
        batch_responses = call_model_batch(
        api_type, 
        prompts, 
        model_name, 
        thinking_list, 
        model, 
        tokenizer, 
        vllm_model, 
        temperature,
        answer_type,
        thinking_end=thinking_end
    )

    
    # Process results
    results = []
    for i, (data, responses, correct_answer) in enumerate(zip(evaluation_data, batch_responses, correct_answers)):
        if responses is None or all(r is None for r in responses):
            continue
        
        # Process each response for this question
        for j, response in enumerate(responses):
            if response is None:
                continue
                
            # Extract the answer
            extracted_answer, raw_thinking, raw_answer = extract_answer(response, answer_only=True)
            
            # Determine if the answer is correct
            is_correct = extracted_answer == correct_answer
            # Prepare the result
            result = {
                'question': data["question"],
                'provided_thinking': data["perturbed_thinking"],
                'raw_thinking': raw_thinking,
                'raw_answer': raw_answer,
                'model_response': response,
                'correct_answer': correct_answer,
                'extracted_answer': extracted_answer,
                'is_correct': is_correct,
                'choices': data["options"],
                'num_tokens' : len(tokenizer.encode(response)),
            }
            
            results.append(result)
    
    return results

def llm_judge_results(results, judge_model, judge_tokenizer, extracted_thinking=False, batch_size=100):
    """
    Process all results using an LLM judge to extract answers.
    Updates results in-place and returns updated statistics.
    """
    logging.info(f"Postprocessing results using LLM judge")
    if "extracted_answer_llm" in results[0]:
        return results

    # Create a dictionary to organize all responses by question
    all_results = {}
    
    # Add responses from main results
    for i, result in enumerate(results):
        question = result["question"]
        if "raw_answer" not in result or result["raw_answer"] is None or len(result["raw_answer"]) == 0:
            continue
        response = result["raw_answer"]
        if extracted_thinking:
            thinking = result["raw_thinking"]
        else:
            thinking = None


        
        all_results[question] = {
            "main_answer": response,
            "main_thinking": thinking,
            "extracted_answer_llm": None,
            "extracted_thinking_answer_llm": None,
            "is_match_thinking_answer_llm": None,
        }
    
    
    # Process responses in batches
    for i in tqdm(range(0, len(all_results), batch_size), desc="Processing batches"):
        batch_items = list(all_results.items())[i:i+batch_size]
        batch_questions = [q for q, _ in batch_items]
        batch_results = [r for _, r in batch_items]
        
        # Extract main answers
        main_responses = [result["main_answer"] for result in batch_results]
        main_thinkings = [result["main_thinking"] for result in batch_results]
        main_questions = [q for q, r in zip(batch_questions, batch_results) if r["main_answer"]]
        extracted_main_answers = extract_answer_llm(
            main_responses,
            judge_model,
            judge_tokenizer,
            main_questions
        )
        # Update with main answers
        for question, answer in extracted_main_answers.items():
            all_results[question]["extracted_answer_llm"] = answer
        if extracted_thinking:
            # extracted_thinking_answers = []
            thinking_answer = extract_answer_llm(
                main_thinkings,
                judge_model,
                judge_tokenizer,
                main_questions
            )
            for question, answer in thinking_answer.items():
                all_results[question]["extracted_thinking_answer_llm"] = answer
        
            for question in all_results:
                if all_results[question]["extracted_answer_llm"] is not None and all_results[question]["extracted_thinking_answer_llm"] is not None:
                    all_results[question]["is_match_thinking_answer_llm"] = all_results[question]["extracted_answer_llm"] == all_results[question]["extracted_thinking_answer_llm"]
                    
    
    # Update main results with extracted answers
    for i, result in enumerate(results):
        question = result["question"]
        if question not in all_results:
            result["extracted_answer_llm"] = None
            result["extracted_thinking_answer_llm"] = None
            result["is_correct_llm"] = False
            result["is_match_thinking_answer_llm"] = False
        else:
            result["extracted_answer_llm"] = all_results[question]["extracted_answer_llm"]
            result["extracted_thinking_answer_llm"] = all_results[question]["extracted_thinking_answer_llm"]

            if "correct_answer" in result and result["extracted_answer_llm"] is not None:
                result["is_correct_llm"] = (result["extracted_answer_llm"] == result["correct_answer"])
            else:
                result["is_correct_llm"] = False
            result["is_match_thinking_answer_llm"] = all_results[question]["is_match_thinking_answer_llm"]
    result["num_llm_judge"] = len(all_results)
    # Return updated results
    return results

def llm_judge_behaviors(results, evaluate_dataset, intervention_key, judge_model, judge_tokenizer, batch_size=100):
    """
    Process all results using an LLM judge to extract behaviors.
    Updates results in-place and returns updated statistics.
    """
    logging.info(f"Postprocessing results using LLM judge") 
    
    # Use judge_behaviors from llm_judge_utils to evaluate responses
    judge_results = judge_behaviors(results, evaluate_dataset, intervention_key, judge_model, judge_tokenizer, batch_size)
    
    # Add judge results to original results
    for i, result in enumerate(results):
        if i < len(judge_results):
            result['judge_evaluation'] = judge_results[i]
    
    # Calculate judge statistics
    judge_stats = {}
    if any('judge_evaluation' in r for r in results):
        # For intervention types like corrupt_option or shift_mapping
        corrected_count = sum(1 for r in results if r.get('judge_evaluation', {}).get('judgment') == 'EXPLICITLY_CORRECTED')
        followed_count = sum(1 for r in results if r.get('judge_evaluation', {}).get('judgment') == 'CONSISTENTLY_FOLLOWED')
        total_evaluated = corrected_count + followed_count
        
        judge_stats = {
            'explicitly_corrected': corrected_count,
            'explicitly_corrected_rate': corrected_count / total_evaluated if total_evaluated > 0 else 0,
            'consistently_followed': followed_count,
            'consistently_followed_rate': followed_count / total_evaluated if total_evaluated > 0 else 0,
        }
    
    return judge_stats

def postprocess_results(results, evaluate_dataset, intervention_key, judge_model=None, judge_tokenizer=None, extracted_thinking=False, batch_size=100, judge_behaviors=False):
    """
    Postprocess results to analyze behavior and apply LLM judging.
    
    Args:
        results: Dictionary with results and config
        intervention_key: Type of intervention used
        judge_model: VLLM model instance for judging responses
        judge_tokenizer: Tokenizer for the judge model
        extracted_thinking: Whether to extract thinking
        batch_size: Batch size for processing
        judge_behaviors: Whether to judge behaviors instead of just extracting answers
        
    Returns:
        Updated results with statistics
    """
    # Compute statistics
    # Standard statistics for single output
    correct_count = sum(1 for r in results["results"] if r.get('is_correct', False))
    accuracy = correct_count / len(results["results"]) if results["results"] else 0
    
    stats = {
        'total_samples': len(results["results"]),
        'correct_samples': correct_count,
        'accuracy': accuracy,
        'avg_num_tokens': sum(result.get('num_tokens', 0) for result in results["results"]) / len(results["results"]) if results["results"] else 0
    }
    
    # Add LLM judge statistics if used
    if judge_behaviors and judge_model is not None and judge_tokenizer is not None:
        logging.info("Using LLM to judge behaviors...")
        # Pass the intervention_key so the judge knows what to look for
        judge_stats = llm_judge_behaviors(
            results["results"], 
            evaluate_dataset,
            intervention_key, 
            judge_model, 
            judge_tokenizer,
            batch_size
        )
        # Add judge statistics to overall stats
        if judge_stats:
            stats['judge_evaluation'] = judge_stats
            
    elif judge_model is not None and judge_tokenizer is not None:
        # Standard answer extraction with LLM judge
        results["results"] = llm_judge_results(
            results["results"], 
            judge_model, 
            judge_tokenizer, 
            extracted_thinking, 
            batch_size
        )
        
        # Standard statistics for single output
        llm_correct_count = sum(1 for r in results["results"] if r.get('is_correct_llm', False))
        llm_accuracy = llm_correct_count / len(results["results"]) if results["results"] else 0
        
        # Update stats
        stats.update({
            'llm_judge_model': results["config"]["judge_model_name"],
            'llm_correct_samples': llm_correct_count,
            'llm_accuracy': llm_accuracy,
            'llm_is_match_thinking_answer': sum(1 for r in results["results"] if r.get('is_match_thinking_answer_llm', False)),
            "num_llm_judge": results["results"][0].get("num_llm_judge", len(results["results"]))
        })

    # Save final results
    results["stats"] = stats
    return results

def main():
    parser = argparse.ArgumentParser(description='Evaluate models on various datasets')
    parser.add_argument('--api_type', type=str, default="vllm",
                      help='Which API to use (vllm,  deepseek)', choices=["vllm", "deepseek"])
    parser.add_argument('--model_name', type=str, required=True,
                      help='Model name or path')
    parser.add_argument('--dataset_type', type=str, default="gpqa",
                      choices=["gpqa", "mmlu"],
                      help='Dataset type to evaluate on')
    parser.add_argument('--output_dir', type=str, default='./results/',
                      help='Output directory for results')
    parser.add_argument('--temperature', type=float, default=0,
                      help='Temperature for API calls')
    parser.add_argument('--dry_run', action='store_true',
                      help='Dry run the evaluation')
    parser.add_argument('--resume', action='store_true',
                      help='Resume from intermediate results')
    # Add batch processing arguments
    parser.add_argument('--batch_size', type=int, default=1,
                      help='Batch size for processing multiple samples at once')
    parser.add_argument('--save_interval', type=int, default=1,
                      help='Save intermediate results every N batches')
    # Add VLLM specific arguments
    parser.add_argument('--tensor_parallel_size', type=int, default=None,
                      help='Tensor parallel size for VLLM')
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.9,
                      help='GPU memory utilization for VLLM')
    parser.add_argument('--eval_only', action='store_true',
                      help='Only evaluate the extracted answer')
    # Add LLM judge parameters
    parser.add_argument('--use_llm_judge', action='store_true',
                      help='Use LLM judge for answer extraction')
    parser.add_argument('--judge_model_name', type=str, default="Qwen/Qwen2.5-32B-Instruct",
                      help='Model to use as judge for answer extraction')
    parser.add_argument('--judge_batch_size', type=int, default=100,
                      help='Batch size for LLM judging')
    # Add faithfulness evaluation parameters
    parser.add_argument('--intervention_types', nargs='+', default=["all"],
                      help='Intervention types to evaluate on')
    parser.add_argument('--benchmark_trace', type=str, default="deepseek_reasoner",
                      help='Benchmark trace to evaluate on')
    # Intra-Draft Faithfulness parameters
    parser.add_argument('--step_locations', nargs='+', default=["first_step", "middle_step", "end_step"],
                      help='Step location to evaluate on Intra-Draft Faithfulness')
    parser.add_argument('--step_types', nargs='+', default=["backtrack", "continue"],
                      help='Step type to evaluate on Intra-Draft Faithfulness')
    # Draft-to-Answer Faithfulness parameters 
    parser.add_argument('--answer_types', nargs='+', default=["standard", "immediate"],
                      help='Answer type to evaluate on Draft-to-Answer Faithfulness')
    args = parser.parse_args()

    # Create output directory
    model_name_short = args.model_name.split("/")[-1]
    args.output_dir = os.path.join(args.output_dir, args.dataset_type, model_name_short)
    args.output_dir = os.path.join(args.output_dir, f"temp_{args.temperature}")
    

    os.makedirs(args.output_dir, exist_ok=True)
                
    # Set API keys from environment
    if args.api_type == "together":
        os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
        if not os.getenv("TOGETHER_API_KEY"):
            raise ValueError("TOGETHER_API_KEY environment variable not set")
    if args.api_type == "deepseek":
        os.environ["DEEPSEEK_API_KEY"] = os.getenv("DEEPSEEK_API_KEY")
        if not os.getenv("DEEPSEEK_API_KEY"):
            raise ValueError("DEEPSEEK_API_KEY environment variable not set")

    # Load dataset
    logging.info(f"Loading {args.dataset_type} dataset...")
    dataset = load_dataset_by_type(args.dataset_type, args.benchmark_trace)
    
    # Initialize models
    model = None
    tokenizer = None
    vllm_model = None


    if "all" in args.intervention_types:
        intervention_types = ["original", "corrupt_option", "shift_mapping", "direct_alternative_conclusion", "plausible_alternative_conclusion"]
    else:
        intervention_types = args.intervention_types
    
    # Load testing model
    if args.api_type == "deepseek":
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-R1")
    else:
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    vllm_model = None
    if not args.eval_only:
        if args.api_type == "vllm":
            logging.info(f"Loading VLLM model {args.model_name} with tensor parallel size {args.tensor_parallel_size}...")
            num_devices = torch.cuda.device_count() 
            vllm_model = LLM(
                model=args.model_name,
                tensor_parallel_size=num_devices if args.tensor_parallel_size is None else args.tensor_parallel_size,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_len=32768,
            )
    judge_model = None
    judge_tokenizer = None
    if args.eval_only:
        logging.info("Using LLM judge to extract answers")
        num_devices = torch.cuda.device_count() 
        judge_model = LLM(
            model=args.judge_model_name,
            tensor_parallel_size=num_devices,  # Use minimal resources for the judge
            gpu_memory_utilization=0.95,
            max_model_len=32768
        )
        judge_tokenizer = AutoTokenizer.from_pretrained(args.judge_model_name)
    
    # Evaluate on all intervention types
    for intervention_type in intervention_types:
        if intervention_type in ["original", "corrupt_option", "shift_mapping"]:
            step_locations = args.step_locations
            step_types = args.step_types
            evaluation_type = "intra_draft"
            answer_types = None
        else:
            step_locations = None
            step_types = None
            evaluation_type = "draft_to_answer"
            answer_types = args.answer_types
        # Load intermediate results if resuming
        if evaluation_type == "intra_draft":
            for step_location in step_locations:
                for step_type in step_types:
                    if step_location not in ["first_step", "middle_step", "end_step"] or step_type not in ["backtrack", "continue"]:
                        logging.error(f"Invalid step location or step type: {step_location} {step_type}")
                        continue
                    if step_type != step_types[0] and intervention_type == "original":
                        continue

                    if intervention_type == "original":
                        intervention_key = f"original_{step_location}"
                    else:
                        intervention_key = f"{step_type}_{intervention_type}_{step_location}"
                    evaluate_subset = dataset.filter(lambda x: x["intervention_type"] == intervention_key)
                    stored_path = os.path.join(args.output_dir, args.benchmark_trace, evaluation_type, f"{intervention_key}.json")
                    saved_questions = []
                    results = {"config": vars(args), "stats": {}, "results": []}
                    if args.resume and os.path.exists(stored_path):
                        logging.info(f"Resuming from {stored_path}")
                        with open(stored_path, 'r') as f:
                            results = json.load(f)
                            saved_questions = [result['question'] for result in results["results"]]
                        if len(evaluate_subset) == len(saved_questions):
                            logging.info(f"No more samples to evaluate for {intervention_key}")
                            if args.eval_only and "original" not in intervention_key:
                                postprocess_results(
                                    results,
                                    evaluate_subset,
                                    intervention_key,
                                    judge_model=judge_model,
                                    judge_tokenizer=judge_tokenizer,
                                    batch_size=args.judge_batch_size,
                                    judge_behaviors=True
                                )
                                with open(stored_path, 'w') as f:
                                    json.dump(results, f, indent=4)
                            elif "original" in intervention_key:
                                results = postprocess_results(
                                    results,
                                    evaluate_subset,
                                    intervention_key,
                                    judge_model=judge_model,
                                    judge_tokenizer=judge_tokenizer,
                                    batch_size=args.judge_batch_size,
                                    extracted_thinking=True,
                                )
                                with open(stored_path, 'w') as f:
                                    json.dump(results, f, indent=4)
                            continue
                    if args.eval_only:
                        raise ValueError("Eval only is supported with completed results")
                    if not os.path.exists(os.path.dirname(stored_path)):
                        os.makedirs(os.path.dirname(stored_path), exist_ok=True)
                    # Process samples in batches
                    logging.info(f"Evaluating {len(dataset)} samples with batch size {args.batch_size} per prompt...")
                    batches = []
                    current_batch = []
                    for idx, sample in enumerate(evaluate_subset):
                        question = sample["question"]
                        # Skip already processed samples
                        if question in saved_questions:
                            continue
                        current_batch.append(sample)
                        if len(current_batch) == args.batch_size or idx == len(evaluate_subset) - 1:
                            batches.append(current_batch)
                            current_batch = []
                    # Process each batch
                    for batch_idx, batch_samples in enumerate(tqdm(batches)):
                        try:
                            batch_results = evaluate(
                                    batch_samples, 
                                    args.api_type, 
                                    args.model_name, 
                                    model=model, 
                                    tokenizer=tokenizer, 
                                    vllm_model=vllm_model, 
                                    temperature=args.temperature, 
                                    thinking_end=False,
                                    dry_run=args.dry_run
                                )
                            results["results"].extend(batch_results)
                            
                            # Save intermediate results at specified intervals
                            if (batch_idx + 1) % args.save_interval == 0:
                                with open(stored_path, 'w') as f:
                                    json.dump(results, f, indent=4)
                                # logging.info(f"Saved intermediate results after {batch_idx + 1} batches ({len(results['results'])} samples)")
                        
                        except Exception as e:
                            logging.error(f"Error processing batch {batch_idx}: {str(e)}")
                            # Save what we have so far
                            with open(stored_path, 'w') as f:
                                json.dump(results, f, indent=4)
                    results = postprocess_results(
                        results,
                        evaluate_subset,
                        intervention_key
                    )
                    with open(stored_path, 'w') as f:
                        json.dump(results, f, indent=4)
        elif evaluation_type == "draft_to_answer":
            for answer_type in answer_types:
                intervention_key = intervention_type
                evaluate_subset = dataset.filter(lambda x: x["intervention_type"] == intervention_key)
                stored_path = os.path.join(args.output_dir, args.benchmark_trace, evaluation_type, f"{intervention_key}_{answer_type}.json")
                saved_questions = []
                results = {"config": vars(args), "stats": {}, "results": []}
                if args.resume and os.path.exists(stored_path):
                    logging.info(f"Resuming from {stored_path}")
                    with open(stored_path, 'r') as f:
                        results = json.load(f)
                        saved_questions = [result['question'] for result in results["results"]]
                if len(evaluate_subset) == len(saved_questions):
                    logging.info(f"No more samples to evaluate for {intervention_key}")
                    if args.eval_only:
                        results = postprocess_results(
                            results,
                            evaluate_subset,
                            intervention_key,
                            judge_model=judge_model,
                            judge_tokenizer=judge_tokenizer,
                            batch_size=args.judge_batch_size,
                            extracted_thinking=True,
                        )
                        with open(stored_path, 'w') as f:
                            json.dump(results, f, indent=4)
                    continue
                if not os.path.exists(os.path.dirname(stored_path)):
                    os.makedirs(os.path.dirname(stored_path), exist_ok=True)
                batches = []
                current_batch = []
                for idx, sample in enumerate(evaluate_subset):
                    question = sample["question"]
                    # Skip already processed samples
                    if question in saved_questions:
                        continue
                    current_batch.append(sample)
                    if len(current_batch) == args.batch_size or idx == len(evaluate_subset) - 1:
                        batches.append(current_batch)
                        current_batch = []
                if args.eval_only:
                    raise ValueError("Eval only is supported with completed results")
                # Process each batch
                for batch_idx, batch_samples in enumerate(tqdm(batches)):
                    try:
                        batch_results = evaluate(
                                batch_samples, 
                                args.api_type, 
                                args.model_name, 
                                model=model, 
                                tokenizer=tokenizer, 
                                vllm_model=vllm_model, 
                                temperature=args.temperature, 
                                answer_type=answer_type,
                                thinking_end=True,
                                dry_run=args.dry_run
                            )
                        results["results"].extend(batch_results)
                        
                        # Save intermediate results at specified intervals
                        if (batch_idx + 1) % args.save_interval == 0:
                            with open(stored_path, 'w') as f:
                                json.dump(results, f, indent=4)
                            # logging.info(f"Saved intermediate results after {batch_idx + 1} batches ({len(results['results'])} samples)")
                    
                    except Exception as e:
                        logging.error(f"Error processing batch {batch_idx}: {str(e)}")
                        # Save what we have so far
                        with open(stored_path, 'w') as f:
                            json.dump(results, f, indent=4)
                    results = postprocess_results(
                        results,
                        evaluate_subset,
                        intervention_key,
                    )
                    with open(stored_path, 'w') as f:
                        json.dump(results, f, indent=4)
    if vllm_model is not None:
        del vllm_model
        torch.cuda.empty_cache()
    if judge_model is not None:
        del judge_model
        torch.cuda.empty_cache()

if __name__ == "__main__":
    main() 