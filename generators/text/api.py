import requests
import re
import logging
from tqdm import tqdm
from torchmetrics.text.bert import BERTScore


logging.basicConfig(level=logging.INFO)

url = "http://localhost:11434/api/chat"

body_enhance = {
    "model": "llama2",
    "messages": [
        {
            "role": "system",
            "content": "Hello, I am an AI assistant specialized in enhancing prompts for realistic and detailed image generation. Please provide a brief description or scenario, and I will enhance it to create a more vivid and detailed prompt suitable for generating high-quality images. For example, you can start with a simple scene like 'a cat sitting on a windowsill' and I'll add more details to it.",
        },
        {"role": "user", "content": "why is the sky blue?"},
    ],
    "stream": False,
}

bertscore = BERTScore()


def create_base_prompt(subject1, verb, subject2):
    return f"A {subject1} is {verb} while a {subject2} is doing [Activity]."


def create_reference_prompt(subject1, verb, subject2):
    return f"A {subject1} is {verb} while a {subject2} is doing an activity"


def enhance_prompt(base_prompt, body_template, model):
    try:
        body = body_template.copy()
        body["model"] = model
        body["messages"][-1][
            "content"
        ] = f'Enhance the following prompt by adding diverse elements while maintaining the core scenario: "{base_prompt}". Focus on diversity in roles, activities, and settings. Output a fully detailed prompt in a string format. ""'
        req = requests.post(url=url, json=body)
        req.raise_for_status()

        prompt_raw = req.json()["message"]["content"]
        prompt = re.search(r'"([^"]+)"', prompt_raw)
        if prompt:
            return prompt.group(1)
        else:
            logging.warning(prompt_raw)
            logging.warning("No prompt found in response")
            return None
    except requests.RequestException as e:
        logging.error(f"Request error: {e}")
        return None
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        return None


def generate(prompt, model="llama2:70b"):
    subject1, verb, subject2 = prompt.split(",")
    base_prompt = create_base_prompt(subject1, verb, subject2)
    reference_prompt = create_reference_prompt(subject1, verb, subject2)
    for _ in range(3):
        prompt = enhance_prompt(base_prompt, body_enhance, model)
        if prompt is not None:
            score = bertscore([prompt], [reference_prompt])["f1"].cpu().numpy()
            print(prompt, score)
            if score > 0.8:
                print("ACCEPTED")
                return prompt
    logging.warning(
        f"Failed to generate an enhanced prompt after 3 attempts for base prompt: {base_prompt}"
    )
    return False
