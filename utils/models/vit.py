import torch
from torch import nn
from transformers import AutoImageProcessor, Swinv2Model
from transformers import logging as transformers_logging


transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()


class Embed(nn.Module):
    def __init__(self, device) -> None:
        super(Clip, self).__init__()
        self.device = device

    def forward(self, image):
        pass


class Swin(Clip):
    def __init__(
        self,
        device,
        model_name="microsoft/swinv2-tiny-patch4-window8-256",
    ):
        super(ViTLargeClip, self).__init__(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)

    def forward(self, image, prompt):
        inputs = self.processor(
            text=[prompt], images=image, return_tensors="pt", padding=True
        ).to(self.device)
        outputs = self.model(**inputs)
        logits_per_image = outputs.logits_per_image
        return float(logits_per_image[0])
