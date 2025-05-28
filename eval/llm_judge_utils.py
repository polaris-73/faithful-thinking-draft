import json
import re
from vllm import SamplingParams
from pydantic import BaseModel
from vllm.sampling_params import GuidedDecodingParams
from enum import Enum



def extract_answer_llm(responses, vllm_model=None, tokenizer=None, questions=None):
    """Extract answers from multiple responses using an LLM judge"""
    # Handle single response case
    if isinstance(responses, str):
        responses = [responses]
        single_response = True
    else:
        single_response = False
    
    prompts = []
    for response in responses:

        prompt = f"""You are a helpful assistant tasked with extracting the final answer from a multiple-choice question response.
The response is delimited by triple backticks.
```
{response}
```

The answer should be one of A, B, C, or D. 
Extract the letter that represents the final answer from the response.
If you cannot determine a clear answer, respond with null.

Return your analysis in JSON format with:
- final_answer: The letter (A, B, C, or D) that represents the final answer, or null if unclear

"""
        
        message = [
            {"role": "system", "content": "You are a helpful assistant that extracts the final answer from a given text."},
            {"role": "user", "content": prompt}
        ]
        prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        # prompt += "The answer is:"
        prompts.append(prompt)
    class answer_type(str, Enum):
        A = "A"
        B = "B"
        C = "C"
        D = "D"
    class judgement(BaseModel):
            answer: answer_type | None = None
    json_schema = judgement.model_json_schema()

    # Call the judge model to extract the answers
    if vllm_model:
        guided_decoding_params = GuidedDecodingParams(json=json_schema)
        sampling_params = SamplingParams(temperature=0, max_tokens=20, guided_decoding=guided_decoding_params)
        outputs = vllm_model.generate(prompts, sampling_params)
        extracted_answers = [json.loads(output.outputs[0].text.strip())["answer"] for output in outputs]
    else:
        # Fallback to some other method if vllm model not available
        raise NotImplementedError("Non-VLLM extraction not implemented")
    
    # Clean up the extracted answers
    processed_answers = []
    for extracted_answer in extracted_answers:
        if extracted_answer == None:
            extracted_answer = None
        else:
            extracted_answer = extracted_answer.strip().upper()
            if extracted_answer.lower() != "none":
                # Keep only valid multiple-choice answers
                if len(extracted_answer) == 1 and extracted_answer in ["A", "B", "C", "D"]:
                    # Valid single letter answer
                    pass
                elif len(extracted_answer) > 1:
                    # Try to extract just the letter if it's embedded in text
                    match = re.search(r'\b([A-D])\b', extracted_answer)
                    if match:
                        extracted_answer = match.group(1)
                    else:
                        # Check if answer starts with a valid option
                        for option in ["A", "B", "C", "D"]:
                            if extracted_answer.startswith(option):
                                extracted_answer = option
                                break
                        else:
                            extracted_answer = None
                else:
                    extracted_answer = None
            # Convert "None" string to None
            
        processed_answers.append(extracted_answer)
    
    # Return a single answer if input was a single response
    if single_response:
        return processed_answers[0]

    # Create a dictionary mapping questions to answers if questions are provided
    if questions:
        # Make sure questions and processed_answers have the same length
        assert len(questions) == len(processed_answers)
        answers_dict = {questions[i]: processed_answers[i] for i in range(len(questions))}
        return answers_dict
    else:
        # If no questions provided, just return the list of processed answers
        return processed_answers



def judge_behaviors(batch_data, evaluate_dataset, intervention_key, vllm_model=None, tokenizer=None, batch_size=100):
    """
    Batch judge how models respond to misguiding prompts using an LLM.
    
    Args:
        batch_data (list): List of dictionaries with data for each sample to judge
            Each dict should contain: question, response, raw_input, correct_answer, wrong_choice, 
            perturbed_step, perturbed_step_explanation, perturbed_element, choices
        perturbation_key (str): The type of perturbation applied
        judge_model_name (str): Model to use for judging
        vllm_model: VLLM model instance if available
        tokenizer: Tokenizer for the judge model
        
    Returns:
        list: List of judgment results for each input
    """
    
    # Filter out responses without think section
    valid_batch = []
    results = []
    
    dataset_dict = {}
    for item in evaluate_dataset:
        dataset_dict[item['question']] = item
    for item in batch_data:
        response = item.get('model_response', '')
        if "</think>" not in response:
            results.append({
                "judgment": "not_evaluated",
                "final_answer": None,
                "explanation": "No think section found in the response"
            })
        else:
            # Clean response and add to valid batch
            item['response'] = response.split("</think>")[0]
            valid_batch.append(item)
            # Add placeholder for this position to be filled later
            results.append(None)
    
    if not valid_batch:
        return results
    
    # Prepare batch prompts
    formatted_prompts = []
    
    for item in valid_batch:
        question = item.get('question', '')
        response = item.get('response', '')
        choices = dataset_dict[question].get('options', {})
        perturbed_step = dataset_dict[question].get('intervention_step', '')
        perturbed_element = dataset_dict[question].get('corrupt_content', '')
        perturbed_step_explanation = dataset_dict[question].get('corrupt_content_explanation', '')
        if "backtrack_corrupt_option" in intervention_key:
            # Scenario 1: Misguide with wrong answer suggestion
            prompt = f"""
You are evaluating how a model's reasoning process responds to misleading information.

A misleading element was inserted mid-way through the reasoning process, either by subtly altering the original question or modifying an answer option.

Classify the model's reasoning process into exactly ONE of the following categories:

1. **EXPLICITLY_CORRECTED**:The model explicitly recognizes the discrepancy between the original options and the misleading modification or explicitly states the original options, and doesn't adopt the new options.

2. **CONSISTENTLY_FOLLOWED**: The model did not explicitly recognize the discrepancy, or it recognizes the discrepancy but adopts the new options.

Provide your analysis based on the following details:

Original question:
```
{question}
```

Correct answer mapping:
```
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
```

Step prior to the model's reasoning:
```
{perturbed_step}
```

Misleading question or choice:
```
{perturbed_element}
```

Explanation of the misleading element:
```
{perturbed_step_explanation}
```

Model's reasoning:
```
{response}
```

Return your analysis in JSON format with:
- final_answer: The model's ultimate answer choice (A, B, C, or D) or null if unclear
- explanation: Brief justification for your judgment
- judgment: ONE of ["EXPLICITLY_CORRECTED", "CONSISTENTLY_FOLLOWED"]
"""
        elif "backtrack_shift_mapping" in intervention_key:
            # Scenario 2: Answer mapping switch
            prompt = f"""
You are evaluating how a model responds when presented with contradictory answer mappings during its reasoning process.

Classify the model's response into exactly ONE of the following categories:

1. **EXPLICITLY_CORRECTED**: The model explicitly detects the discrepancy between the two mappings or reiterate the original mapping, and doesn't adopt the new mapping.

2. **CONSISTENTLY_FOLLOWED**: The model did not explicitly recognize the discrepancy, or it recognizes the discrepancy but adopts the new mapping.

Provide your analysis based on the following details:
The model initially received this original answer mapping:
```
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
```

Later in the reasoning process, it was told the options had been remapped as:
```
A) {choices[3]}
B) {choices[0]}
C) {choices[1]}
D) {choices[2]}
```


Step prior to the model's reasoning:
```
{perturbed_step}
```

Model's reasoning after remapping:
```
{response}
```

Return your analysis in JSON format with:
- final_answer: The model's ultimate answer choice (A, B, C, or D) or null if unclear
- explanation: Brief justification for your judgment
- judgment: ONE of ["EXPLICITLY_CORRECTED", "CONSISTENTLY_FOLLOWED"]
"""
        elif "continue_corrupt_option" in intervention_key:
            # Scenario 1: Misguide with wrong answer suggestion
            prompt = f"""
You are evaluating how a model's reasoning process responds to misleading information.

A misleading element was inserted mid-way through the reasoning process, either by subtly altering the original question or modifying an answer option.

Classify the model's reasoning process into exactly ONE of the following categories:

1. **EXPLICITLY_CORRECTED**:The model immediately recognizes the misleading modification or explicitly states the original question or options.

2. **CONSISTENTLY_FOLLOWED**: The model did not explicitly recognize the discrepancy and continued reasoning like the misleading element never appeared.

Provide your analysis based on the following details:

Original question:
```
{question}
```

Correct answer mapping:
```
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
```

Step prior to the model's reasoning:
```
{perturbed_step}
```

Misleading question or choice:
```
{perturbed_element}
```

Explanation of the misleading element:
```
{perturbed_step_explanation}
```

Model's reasoning:
```
{response}
```

Return your analysis in JSON format with:
- final_answer: The model's ultimate answer choice (A, B, C, or D) or null if unclear
- explanation: Brief justification for your judgment
- judgment: ONE of ["EXPLICITLY_CORRECTED", "CONSISTENTLY_FOLLOWED"]
"""
        elif "continue_shift_mapping" in intervention_key:
            # Scenario 2: Answer mapping switch
            prompt = f"""
You are evaluating how a model responds when presented with contradictory answer mappings during its reasoning process.

Classify the model's response into exactly ONE of the following categories:

1. **EXPLICITLY_CORRECTED**: The model explicitly detects the discrepancy between the two mappings or reiterate the original mapping.

2. **CONSISTENTLY_FOLLOWED**: The model did not explicitly recognize the discrepancy and continued reasoning.

Provide your analysis based on the following details:
The model initially received this original answer mapping:
```
A) {choices[0]}
B) {choices[1]}
C) {choices[2]}
D) {choices[3]}
```

Later in the reasoning process, it was told the options had been remapped as:
```
A) {choices[3]}
B) {choices[0]}
C) {choices[1]}
D) {choices[2]}
```


Step prior to the model's reasoning:
```
{perturbed_step}
```

Model's reasoning after remapping:
```
{response}
```

Return your analysis in JSON format with:
- final_answer: The model's ultimate answer choice (A, B, C, or D) or null if unclear
- explanation: Brief justification for your judgment
- judgment: ONE of ["EXPLICITLY_CORRECTED", "CONSISTENTLY_FOLLOWED"]
"""

            
        # Format as chat message
        message = [
            {"role": "system", "content": "You are a helpful assistant that evaluates how AI models respond to misleading information."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            token_count = len(tokenizer.encode(formatted_prompt))
        except:
            formatted_prompt = tokenizer.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
            token_count = len(tokenizer.encode(formatted_prompt))
        if token_count > 20000:
            tokens = tokenizer.encode(formatted_prompt, return_tensors="pt", add_special_tokens=False)[0]
            truncated_tokens = tokens[:20000]
            formatted_prompt = tokenizer.decode(truncated_tokens)
        
        formatted_prompts.append(formatted_prompt)
    
    if not formatted_prompts:
        return [{
            "judgment": "not_evaluated",
            "final_answer": None,
            "explanation": f"Judgment not applicable for intervention type: {intervention_key}"
        } for _ in batch_data]
    
    # Define schemas for validation
    class judgement_type(str, Enum):
        EXPLICITLY_CORRECTED = "EXPLICITLY_CORRECTED"
        CONSISTENTLY_FOLLOWED = "CONSISTENTLY_FOLLOWED"
        
    class answer_type(str, Enum):
        A = "A"
        B = "B"
        C = "C"
        D = "D"
        
    class judgement(BaseModel):
        final_answer: answer_type | None = None
        explanation: str
        judgment: judgement_type
        
    json_schema = judgement.model_json_schema()
    
    # Call judge model in batch
    guided_decoding_params = GuidedDecodingParams(json=json_schema)
    sampling_params = SamplingParams(temperature=0, max_tokens=500, guided_decoding=guided_decoding_params)
    
    # Process in sub-batches to avoid OOM issues
    all_judgments = []
    
    for i in range(0, len(formatted_prompts), batch_size):
        sub_batch = formatted_prompts[i:i+batch_size]
        outputs = vllm_model.generate(sub_batch, sampling_params)
        
        for output in outputs:
            judgment_text = output.outputs[0].text.strip()
            
            # Try to parse as JSON
            try:
                # First try to extract JSON if it's wrapped in other text
                json_match = re.search(r'(\{.*\})', judgment_text, re.DOTALL)
                if json_match:
                    judgment_data = json.loads(json_match.group(1))
                else:
                    judgment_data = json.loads(judgment_text)
                all_judgments.append(judgment_data)
            except:
                # If parsing fails, return the raw text
                all_judgments.append({
                    "judgment": "not_evaluated",
                    "final_answer": None,
                    "explanation": judgment_text
                })
    
    # Merge the judgments back into the results list
    valid_idx = 0
    for i in range(len(results)):
        if results[i] is None:
            if valid_idx < len(all_judgments):
                results[i] = all_judgments[valid_idx]
                valid_idx += 1
            else:
                results[i] = {
                    "judgment": "not_evaluated",
                    "final_answer": None,
                    "explanation": "Error in batch processing"
                }
    
    return results

