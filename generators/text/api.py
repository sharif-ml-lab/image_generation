import re
import random
import requests
import logging
from tqdm import tqdm
from torchmetrics.text.bert import BERTScore


logging.basicConfig(level=logging.INFO)

url = "http://localhost:11434/api/chat"
bertscore = BERTScore()


def summarize_prompt(maintain_prompt, text, max_lengh, model):
    summarize_request = f'Summarize the following prompt "{text}" in {max_lengh} max characters by keeping this main scenario ({maintain_prompt}). Output the final result prompt in a string format enclosed within double quotation marks.'
    body = {
        "model": model,
        "options": {"temperature": 0.65},
        "messages": [
            {
                "role": "system",
                "content": "Hello, I am an AI assistant specialized in summarizing prompt which keep much information from the input as much as possible.",
            },
            {"role": "user", "content": summarize_request},
        ],
        "stream": False,
    }
    for attempt in range(5):
        try:
            req = requests.post(url=url, json=body)
            req.raise_for_status()

            prompt_raw = req.json()["message"]["content"]
            prompt = re.search(r'"([^"]+)"', prompt_raw)

            if prompt and len(prompt.group(1).strip()) > 200:
                enhanced_prompt = prompt.group(1)
                return enhanced_prompt
            else:
                body["options"]["temperature"] += 0.03
                logging.warning("Bad/Short Output")
        except requests.RequestException as e:
            logging.error(f"Request error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

    logging.warning(
        f"Failed to generate a sufficiently summarized prompt after 3 attempts."
    )
    return None


def summarize(maintain_prompt, text, max_lengh=460, model="llama2:70b"):
    for _ in range(3):
        prompt = summarize_prompt(maintain_prompt, text, max_lengh, model)
        if prompt is not None:
            return prompt
        else:
            logging.warning("Try Again")
    logging.warning(
        f"Failed to generate an summarized prompt after 3 attempts for base prompt: {maintain_prompt}"
    )
    return False


def enhance_prompt(
    base_prompt, previous_prompts, model, combination, similarity_threshold=0.975
):
    base_scenario = f"Include vivid, sharp details for a man with skin color ({combination[0]}), attire ({combination[1]}), and age ({combination[2]}) and a woman with skin color ({combination[3]}), attire ({combination[4]}), and age ({combination[5]}). Reflect their financial situations ({combination[6]}) in the context of the scene. Emphasize clear, realistic representations of their expressions, the textures of their clothes, and the surrounding environment. Prioritize natural lighting and lifelike details to bring the scene to life."
    if "[Single Woman]" in base_prompt:
        base_scenario = f"Focus on a woman with skin color ({combination[3]}), attire ({combination[4]}), and age ({combination[5]}), representing her financial situation ({combination[6]}). Ensure the description vividly captures her expression, the fabric of her clothing, and her setting in high definition. Emphasize natural lighting and realistic, clear details to enrich the scene's authenticity."
    if "[Single Man]" in base_prompt:
        base_scenario = f"Detail a man with skin color ({combination[0]}), attire ({combination[1]}), and age ({combination[2]}), considering his financial status ({combination[6]}). Describe with clarity his facial features, attire texture, and environment. Use realistic, sharp descriptions to highlight natural lighting and the scene's lifelike qualities."
    enhancement_request = f"Enhance '{base_prompt}' by integrating: {base_scenario} Maintain the core scenario, focusing on creating a deeply immersive, fully realized, ultra-detailed image. Limit to 470 characters."
    body = {
        "model": model,
        "options": {"temperature": 0.61},
        "messages": [
            {
                "role": "system",
                "content": "Hello, I am an AI assistant specialized in enhancing prompts for creating highly realistic, detailed, and vivid images. Provide a scenario, and I'll refine it to maximize clarity and lifelike details for image generation. Start with a simple scene for detailed enhancement."
            },
            {"role": "user", "content": enhancement_request},
        ],
        "stream": False,
    }
    for attempt in range(4):
        try:
            req = requests.post(url=url, json=body)
            req.raise_for_status()

            prompt_raw = req.json()["message"]["content"]
            prompt = re.search(r'"([^"]+)"', prompt_raw)

            if prompt and len(prompt.group(1).strip()) > 200:
                enhanced_prompt = prompt.group(1)
                for prev_prompt in previous_prompts:
                    similarity = (
                        bertscore([enhanced_prompt], [prev_prompt])["f1"].cpu().numpy()
                    )
                    print(similarity)
                    if similarity > similarity_threshold:
                        body["options"]["temperature"] += 0.02
                        logging.warning("Similar Prompt")
                        break
                else:
                    return enhanced_prompt
            else:
                logging.warning("Bad Output")
        except requests.RequestException as e:
            logging.error(f"Request error: {e}")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")

    logging.warning(
        f"Failed to generate a sufficiently diverse prompt after 3 attempts."
    )
    return None


def generate(base_prompt, previous_prompts, combination, model="llama2:70b"):
    for _ in range(3):
        prompt = enhance_prompt(base_prompt, previous_prompts, model, combination)
        if prompt is not None:
            return prompt
        else:
            logging.warning("Try Again")
    logging.warning(
        f"Failed to generate an enhanced prompt after 3 attempts for base prompt: {base_prompt}"
    )
    return False
