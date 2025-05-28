import os
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential
from vllm import SamplingParams

def add_thinking_template(text, model_name, tokenizer, thinking=None, thinking_end=True, immediate_answer=False):
    text = [{
        "role": "user",
        "content": text
    }]
    text = tokenizer.apply_chat_template(text, tokenize=False)

    if "deepseek" in model_name.lower() or "DeepScaleR" in model_name or "skywork" in model_name.lower():
        if thinking is None:
            return text+"<\uff5cAssistant\uff5c><think>\\n"
        else:
            prompt = text+"<\uff5cAssistant\uff5c>"
            if "<think>\\n" not in thinking:
                prompt += "<think>\\n"
            prompt += thinking
            prompt = prompt.replace("</think>", "")
            if thinking_end:
                prompt += "\n</think>\n\n"
            # return prompt
    elif "qwq" in model_name.lower() or "qwen3" in model_name.lower():
        if thinking is None:
            return text+"<|im_start|>assistant\\n<think>\\n"
        else:
            prompt = text+"<|im_start|>assistant\\n"
            if "<think>\\n" not in thinking:
                prompt += "<think>\\n"
            prompt += thinking
            prompt = prompt.replace("</think>", "")
            if thinking_end:
                prompt += "\n</think>\n\n"
            # return prompt
    else:
        raise ValueError(f"Model {model_name} not supported")
    if immediate_answer:
        prompt += " The answer is:"
    
    return prompt

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def call_deepseek_api(formatted_prompt, model_name, thinking=None, temperature=0, thinking_end=True, immediate_answer=False):
    """Call DeepSeek API"""
    client = OpenAI(base_url="https://api.deepseek.com/beta", api_key=os.getenv("DEEPSEEK_API_KEY"))
    
    # Only request multiple outputs if temperature > 0
    if thinking is not None:
        if "<think>\\n" not in thinking:
            thinking = "<think>\\n" + thinking 
        if "</think>" not in thinking and thinking_end:
            thinking = thinking + "</think>"
            if immediate_answer:
                thinking += " The answer is:"
        messages = [{"role": "user", "content": formatted_prompt}, {"role": "assistant", "content": thinking, "prefix": True}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            stop=['```'],
            n=1
        )
    else:
    
        messages = [{"role": "user", "content": formatted_prompt}]
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            n=1
        )
    # Return a list of responses
    if thinking is not None:
        if response.choices[0].message.reasoning_content is None:
            return [response.choices[0].message.content]
        else:
            return ["<think>\\n" + response.choices[0].message.reasoning_content + "</think>" + response.choices[0].message.content]
    else:
        return ["<think>\\n" + response.choices[0].message.reasoning_content + "</think>" + response.choices[0].message.content]


def call_vllm_batch(prompts, model_name, vllm_model, tokenizer, thinking_list=None, temperature=0, thinking_end=True, immediate_answer=False):
    """Call VLLM with batch processing and multiple outputs per prompt""" 
    # Prepare messages for each prompt
    all_messages = []
    all_message_tokens = []
    for i, prompt in enumerate(prompts):
        thinking = thinking_list[i] if thinking_list and i < len(thinking_list) else None
        messages = add_thinking_template(prompt, model_name, tokenizer, thinking, thinking_end, immediate_answer)
        all_messages.append(messages)
        all_message_tokens.append(len(tokenizer.encode(messages)))
    if len(prompts) == 1:
        all_messages = all_messages[0]
        all_message_tokens = all_message_tokens[0]
    # Generate responses for all prompts
    if all_message_tokens > 16384:
        max_tokens = 32768
    else:
        max_tokens = 16384
    if temperature > 0:
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.95,
        )
    else:
        sampling_params = SamplingParams(
            n=1,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    all_outputs = vllm_model.generate(all_messages, sampling_params)
    
    # Process outputs for each prompt
    batch_responses = []
    for outputs in all_outputs:
        prompt_responses = [output.text for output in outputs.outputs]
        batch_responses.append(prompt_responses)
    
    return batch_responses