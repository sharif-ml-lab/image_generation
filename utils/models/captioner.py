import torch
from torch import nn
from transformers import (
    AutoTokenizer,
    ViTImageProcessor,
    BlipProcessor,
    BlipForConditionalGeneration,
    VisionEncoderDecoderModel,
)


class Captioner(nn.Module):
    def __init__(self, device, half_precision=False) -> None:
        super(Captioner, self).__init__()
        self.device = device
        self.dtype = torch.float16 if half_precision else torch.float32

    def forward(self, image):
        pass


class BLIPCaptioner(Captioner):
    def __init__(
        self,
        device,
        half_precision=False,
        model_name="Salesforce/blip-image-captioning-base",
    ):
        super(BLIPCaptioner, self).__init__(device, half_precision)
        self.processor = BlipProcessor.from_pretrained(model_name)
        self.model = BlipForConditionalGeneration.from_pretrained(
            model_name, torch_dtype=self.dtype
        ).to(self.device)

    def forward(self, image, caption=None):
        inputs = self.processor(text=caption, images=image, return_tensors="pt").to(
            self.device, self.dtype
        )
        outputs = self.model.generate(**inputs, max_new_tokens=20)
        return self.processor.batch_decode(outputs, skip_special_tokens=True)


class ViTGPT2Captioner(Captioner):
    def __init__(self, device, model_name="nlpconnect/vit-gpt2-image-captioning"):
        super(ViTGPT2Captioner, self).__init__(device, False)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name).to(
            self.device
        )

    def forward(self, image):
        inputs = self.feature_extractor(
            images=image, return_tensors="pt"
        ).pixel_values.to(self.device)
        outputs = self.model.generate(inputs, {"max_new_tokens": 20, "num_beams": 4})
        return self.tokenizer.batch_decode(outputs, skip_special_tokens=True)