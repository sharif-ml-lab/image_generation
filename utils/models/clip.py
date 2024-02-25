import torch
from torch import nn
import open_clip
from transformers import CLIPProcessor, CLIPModel
from transformers import AlignProcessor, AlignModel
from transformers import AltCLIPModel, AltCLIPProcessor
from transformers import logging as transformers_logging


transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()


class Clip(nn.Module):
    def __init__(self, device) -> None:
        super(Clip, self).__init__()
        self.device = device

    def forward(self, image):
        pass


class AltClip(Clip):
    def __init__(self, device, model_name="BAAI/AltCLIP"):
        super(AltClip, self).__init__(device)
        self.model = AltCLIPModel.from_pretrained(model_name).to(device)
        self.processor = AltCLIPProcessor.from_pretrained(model_name)

    def forward(self, image, text_list):
        with torch.no_grad():
            inputs = self.processor(
                text=text_list, images=image, return_tensors="pt", padding=True
            ).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
        return logits_per_image.softmax(dim=1).cpu().numpy()


class AlignClip(Clip):
    def __init__(self, device, model_name="kakaobrain/align-base"):
        super(AltClip, self).__init__(device) 
        self.model = AlignModel.from_pretrained(model_name).to(device)
        self.processor = AlignProcessor.from_pretrained(model_name)

    def forward(self, image, text_list):
        with torch.no_grad():
            inputs = self.processor(text=candidate_labels, images=image, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
        return logits_per_image.softmax(dim=1).cpu().numpy()


class ViTOpenAIClip(Clip):
    def __init__(
        self,
        device,
        base_name="ViT-B-32",
        pretrained="laion2b_s34b_b79k",
    ):
        super(ViTOpenAIClip, self).__init__(device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name=base_name, pretrained=pretrained
        )
        self.model = self.model.to(device)
        self.tokenizer = open_clip.get_tokenizer(base_name)

    def forward(self, image, text_list):
        image = self.preprocess(image).unsqueeze(0).to(self.device)
        text = self.tokenizer(text_list).to(self.device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return text_probs.cpu().numpy()


class EvaClip(Clip):
    def __init__(
        self,
        device,
        model_name="EVA02-CLIP-L-14",
    ):
        super(EvaClip, self).__init__(device)
        self.model, _, self.preprocess = create_model_and_transforms(
            model_name, "eva_clip", force_custom_clip=True
        )
        self.model = self.model.to(device)
        self.tokenizer = get_tokenizer(model_name)

    def forward(self, image, text):
        image = self.preprocess(image).unsqueeze(0).to(device)
        text = self.tokenizer([text]).to(device)
        with torch.no_grad():
            image_features = self.model.encode_image(image)
            text_features = self.model.encode_text(text)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        return float(text_probs[0])
