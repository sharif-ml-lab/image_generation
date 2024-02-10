import torch
from torch import nn
from transformers import CLIPProcessor, CLIPModel
from transformers import logging as transformers_logging


transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()


class Clip(nn.Module):
    def __init__(self, device) -> None:
        super(Clip, self).__init__()
        self.device = device

    def forward(self, image):
        pass


class ViTLargeClip(Clip):
    def __init__(
        self,
        device,
        model_name="openai/clip-vit-base-patch32",
    ):
        super(ViTLargeClip, self).__init__(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

    def forward(self, image, prompt):
        try:
            inputs = self.processor(
                text=[prompt], images=image, return_tensors="pt", padding=True
            ).to(self.device)
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            return float(logits_per_image[0])
        except:
            return None # Clip Prompt Max Token Limit