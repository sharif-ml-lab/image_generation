import torch
import numpy as np
import torchvision.transforms as transforms
from transformers import (
    ViltProcessor,
    BlipProcessor,
    Blip2ForConditionalGeneration,
    BlipForQuestionAnswering,
    ViltForQuestionAnswering,
)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def vqa_vilt_base(image, text):
    processor = ViltProcessor.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
    model = ViltForQuestionAnswering.from_pretrained(
        "dandelin/vilt-b32-finetuned-vqa"
    ).to(DEVICE)
    encoding = processor(image, text, return_tensors="pt").to(DEVICE)
    outputs = model(**encoding)
    logits = outputs.logits
    idx = logits.argmax(-1).item()
    return model.config.id2label[idx]


def vqa_blip(image, text, model):
    MODEL_CHECKPOINT = {
        "flan_t5_xl": "Salesforce/blip2-flan-t5-xl",
        "flan_t5_xxl": "Salesforce/blip2-flan-t5-xxl",
        "capfilt_large": "Salesforce/blip-vqa-capfilt-large",
    }
    processor = BlipProcessor.from_pretrained(MODEL_CHECKPOINT[model])
    if "flan" in model:
        model = Blip2ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT[model])
    else:
        model = BlipForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT[model])
    model = model.to(DEVICE)
    inputs = processor(image, text, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)


def vqa_alignment_metric(loader_fake, model="vlit", subject="man", relation="driving"):
    scores = []
    questions = [
        [f"Who is {relation}?", subject],
        [f"The man is {relation} or the woman?", subject]
    ]
    for fake_batch in loader_fake:
        pill_fake = transforms.ToPILImage()(fake_batch[0])
        total = 0
        for text, expected in questions:
            if model == "vlit":
                answer = vqa_vilt_base(pill_fake, text)
            elif model == "git":
                answer = vqa_git_large(pill_fake, text)
            else:
                answer = vqa_blip(pill_fake, text, model=model)
            total += int(expected in answer.split())
        scores.append(total / 2)
    scores_array = np.array(scores)
    return scores_array.mean(), scores_array.std()
