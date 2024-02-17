import requests
import re
from tqdm import tqdm


# Using Ollama Library
url = "http://localhost:11434/api/chat"
body = {
    "model": "llama2",
    "messages": [
        {
            "role": "system",
            "content": "Hello. I am a promopt generator and enhancmentor for text-to-image models wich generate fully-detailed prompt to have realistic image. How can I help you?",
        },
        {"role": "user", "content": "why is the sky blue?"},
    ],
    "stream": False,
}
body_enahnce = {
    "model": "llama2",
    "messages": [
        {
            "role": "system",
            "content": "Hello. I am a promopt improver for text-to-image models to have realistic image. Give me the prompt i will send you the enhacned prompt",
        },
        {"role": "user", "content": "why is the sky blue?"},
    ],
    "stream": False,
}


def enhance_prompt(prompt):
    body_enahnce["messages"][-1][
        "content"
    ] = f"Enhance the following prompt to have better better image as result: '{prompt}'. please just output prompts in string format like \"\""
    req = requests.post(url=url, json=body_enahnce)
    prompt_raw = req.json()["message"]["content"]
    prompt = re.findall('".+"', prompt_raw)[0].replace('"', "").strip()
    return prompt


def generate_full_prompt(base_prompt, body):
    body["messages"][-1][
        "content"
    ] = f'Provide a list of fully detailed prompt that emphasize on describing this scnece \'{base_prompt}\' please make sure the woman and man exists in prompt. keep the semantic information but diverse the scenes, objects, dresses, filling, faces, skin color, ages, positions, financial situation, weather, personality, and other features. please just output prompts in list of string like ["", "", ...]'
    req = requests.post(url=url, json=body)
    prompt_raw = req.json()["message"]["content"]
    prompts = re.findall('\d+\. ".+"', prompt_raw)
    prompts = list(
        map(lambda txt: re.findall('".+"', txt)[0].replace('"', "").strip(), prompts)
    )
    return prompts


def generate(base_prompt, model="llama2:70b"):
    body["model"] = model
    body_enahnce["model"] = model
    prompts_list = generate_full_prompt(base_prompt, body)
    enhanced_prompts = []
    for prompt in tqdm(prompts_list, desc="Prompt Enhancement"):
        for _ in range(3):
            try:
                enhanced = enhance_prompt(prompt)
                enhanced_prompts.append(enhanced)
                break
            except Exception as e:
                print(str(e))
                continue
    return enhanced_prompts
