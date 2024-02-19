import requests
import re
from tqdm import tqdm
from torchmetrics.text.bert import BERTScore


bertscore = BERTScore()

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
    ] = f"Enhance the following prompt (max-length 370 character) to have better better image as result: '{prompt}'. please just output prompts in string format like \"\""
    req = requests.post(url=url, json=body_enahnce)
    prompt_raw = req.json()["message"]["content"]
    prompt = re.findall('".+"', prompt_raw)[0].replace('"', "").strip()
    return prompt


def generate_full_prompt(base_prompt, body):
    body["messages"][-1][
        "content"
    ] = f'Provide 2 or 3 fully detailed prompts that emphasize on describing this scnece \'{base_prompt}\' please make sure the woman and man exists in prompt. keep the semantic information constant but diverse the scenes, objects, dresses, filling, faces, skin color, ages, positions, financial situation, weather, personality, and other features. please just output prompts in list of string like ["", "", ...] (max-length of each prompt 370 character)'
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
    for _ in range(3):
        try:
            prompts_list = generate_full_prompt(base_prompt, body)
            best_prompt = max(prompts_list, key=lambda text: bertscore([text], [base_prompt])["f1"].cpu().numpy())
            break
        except Exception as e:
            print(str(e))
            continue
    for _ in range(3):
        try:
            enhanced = enhance_prompt(best_prompt)
            return enhanced
        except Exception as e:
            print(str(e))
            continue
    return False