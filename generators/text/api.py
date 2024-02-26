import re
import random
import requests
import logging
from tqdm import tqdm
from torchmetrics.text.bert import BERTScore


logging.basicConfig(level=logging.INFO)

url = "http://localhost:11434/api/chat"
bertscore = BERTScore()


def generate_non_stereotypical_prompt(
    base_prompt, previous_prompts, combination, model="llama2:70b"
):
    enhanced_prompt = enhance_prompt(base_prompt, previous_prompts, model, combination)
    if enhanced_prompt:
        print("Generated Prompt Length:", len(enhanced_prompt))
        return enhanced_prompt
    else:
        logging.warning(
            f"Unable to generate an enhanced prompt for combination {combination} and base prompt: {base_prompt}"
        )


def enhance_prompt(
    base_prompt, previous_prompts, model, combination, similarity_threshold=0.975
):
    enhancement_request = f'Enhance the following prompt by adding elements of diversity in skin color for a man ({combination[0]}) with dress type ({combination[1]}) and age ({combination[2]}) and also a woman ({combination[3]}), with dress type ({combination[4]}) and age ({combination[5]}), financial situations ({combination[6]}), area ({combination[7]}), **while emphasizing on maintaining the core scenario**: **"{base_prompt.replace("[Activity]", combination[-1])}"**. Include fully-details that vividly depict the real environment and the individuals circumstances. Output the final result prompt (max-length 470 characters) in a string format enclosed within double quotation marks.'
    body = {
        "model": model,
        "options": {"temperature": 0.64},
        "messages": [
            {
                "role": "system",
                "content": "Hello, I am an AI assistant specialized in enhancing different, diversed and variation of prompts for realistic and detailed image generation. Please provide a brief description or scenario, and I will enhance it to create a more vivid and detailed prompt (max-length 470 characters) suitable for generating high-quality images. For example, you can start with a simple scene like 'a cat sitting on a windowsill' and I'll add more details to it. ",
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
                        body["options"]["temperature"] += 0.03
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
        prompt = generate_non_stereotypical_prompt(
            base_prompt, previous_prompts, combination, model
        )
        if prompt is not None:
            return prompt
        else:
            logging.warning("Try Again")
    logging.warning(
        f"Failed to generate an enhanced prompt after 3 attempts for base prompt: {base_prompt}"
    )
    return False
