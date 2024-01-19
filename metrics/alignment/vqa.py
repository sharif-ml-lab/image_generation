import torch
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
    if 'flan' in model:
        model = Blip2ForConditionalGeneration.from_pretrained(MODEL_CHECKPOINT[model])
    else:
        model = BlipForQuestionAnswering.from_pretrained(MODEL_CHECKPOINT[model])
    model = model.to(DEVICE)
    inputs = processor(image, text, return_tensors="pt").to(DEVICE)
    output = model.generate(**inputs)
    return processor.decode(output[0], skip_special_tokens=True)


def vqa_alignment_metric(loader_fake, model="vlit"):
    responses = []
    questions = [
        "What activity is the woman in the image doing?",
        "What activity is the man in the image doing?",
        "The man is driving or the woman?",
        "Is the image realistic or fake?",
        "What is the color of car?",
    ]
    for fake_batch in loader_fake:
        responses.append([])
        pill_fake = transforms.ToPILImage()(fake_batch[0])
        for text in questions:
            if model == "vlit":
                answer = vqa_vilt_base(pill_fake, text)
            elif model == "git":
                answer = vqa_git_large(pill_fake, text)
            else:
                answer = vqa_blip(pill_fake, text, model=model)
            responses[-1].append(text + ' ' + answer)
    return responses
