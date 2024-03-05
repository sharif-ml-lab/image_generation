import torch
from torch import nn
from transformers import (
    ViltProcessor,
    BlipProcessor,
    Blip2ForConditionalGeneration,
    BlipForQuestionAnswering,
    ViltForQuestionAnswering,
)
from transformers import logging as transformers_logging


transformers_logging.disable_progress_bar()
transformers_logging.set_verbosity_error()


class VQA(nn.Module):
    def __init__(self, device) -> None:
        super(VQA, self).__init__()
        self.device = device

    def forward(self, image):
        pass


class VlitBaseVQA(VQA):
    def __init__(
        self,
        device,
        model_name="dandelin/vilt-b32-finetuned-vqa",
    ):
        transformers_logging.set_verbosity_error()
        super(VlitBaseVQA, self).__init__(device)
        self.processor = ViltProcessor.from_pretrained(model_name)
        self.model = ViltForQuestionAnswering.from_pretrained(model_name).to(
            self.device
        )

    def forward(self, image, text):
        inputs = self.processor(image, text, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)
        logits = outputs.logits
        idx = logits.argmax(-1).item()
        return self.model.config.id2label[idx]


class FlanXlVQA(VQA):
    def __init__(
        self,
        device,
        model_name="Salesforce/blip2-flan-t5-xl",
    ):
        super(FlanXlVQA, self).__init__(device)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

    def forward(self, image, text):
        inputs = self.processor(image, text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        return self.processor.decode(outputs[0], skip_special_tokens=True)


class FlanSuperXlVQA(VQA):
    def __init__(
        self,
        device,
        model_name="Salesforce/blip2-flan-t5-xll",
    ):
        super(FlanXlVQA, self).__init__(device)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_name).to(
            self.device
        )

    def forward(self, image, text):
        inputs = self.processor(image, text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        return self.processor.decode(outputs[0], skip_special_tokens=True)


class CapFiltVQA(VQA):
    def __init__(
        self,
        device,
        model_name="Salesforce/blip-vqa-capfilt-large",
    ):
        super(FlanXlVQA, self).__init__(device)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForQuestionAnswering.from_pretrained(model_name).to(
            self.device
        )

    def forward(self, image, text):
        inputs = self.processor(image, text, return_tensors="pt").to(self.device)
        outputs = self.model.generate(**inputs)
        return self.processor.decode(outputs[0], skip_special_tokens=True)
