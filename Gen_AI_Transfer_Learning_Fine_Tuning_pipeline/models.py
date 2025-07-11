from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM, GPT2LMHeadModel, GPT2Tokenizer,
    LayoutLMv3Processor, LayoutLMv3ForTokenClassification,
    ViTForImageClassification, ViTImageProcessor
)
import torch
from PIL import Image

class BaseModel:
    def infer(self, *args, **kwargs):
        raise NotImplementedError

class BartModel(BaseModel):
    def __init__(self, model_name):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def infer(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        output_ids = self.model.generate(**inputs)
        return self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

class GPT2Model(BaseModel):
    def __init__(self, model_name):
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

    def infer(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        outputs = self.model.generate(**inputs, max_length=100)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

class LayoutLMv3Model(BaseModel):
    def __init__(self, model_name):
        self.processor = LayoutLMv3Processor.from_pretrained(model_name)
        self.model = LayoutLMv3ForTokenClassification.from_pretrained(model_name)

    def infer(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        return outputs.logits.argmax(-1).tolist()

class ViTModel(BaseModel):
    def __init__(self, model_name):
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.model = ViTForImageClassification.from_pretrained(model_name)

    def infer(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt")
        outputs = self.model(**inputs)
        predicted = torch.argmax(outputs.logits, dim=1)
        return self.model.config.id2label[predicted.item()]
