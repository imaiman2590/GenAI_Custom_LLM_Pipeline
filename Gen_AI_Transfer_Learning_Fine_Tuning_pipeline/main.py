from fastapi import FastAPI, UploadFile, File, Form
from metrics import DOC_SUCCESS, DOC_FAILURE, start_metrics_server
from utils import extract_text_from_pdf, extract_text_from_docx, load_image
from models import BartModel, GPT2Model, LayoutLMv3Model, ViTModel
from training_pipeline import fine_tune_text_model
from config import MODEL_TYPES
import pandas as pd
from io import BytesIO

app = FastAPI()
start_metrics_server()

# Initialize with default models - can be swapped after fine-tuning
models = {
    "bart": BartModel(MODEL_TYPES["bart"]),
    "gpt2": GPT2Model(MODEL_TYPES["gpt2"]),
    "layoutlmv3": LayoutLMv3Model(MODEL_TYPES["layoutlmv3"]),
    "vit": ViTModel(MODEL_TYPES["vit"]),
}

@app.post("/infer/")
async def infer(
    input_type: str = Form(...),
    model_type: str = Form(...),
    file: UploadFile = File(None),
    prompt: str = Form(None)
):
    if model_type not in models:
        return {"error": f"Model type '{model_type}' not supported."}
    model = models[model_type]

    if input_type == "document":
        if file is None:
            return {"error": "File required for document input."}
        content = await file.read()
        try:
            if file.filename.endswith(".pdf"):
                text = extract_text_from_pdf(content)
            elif file.filename.endswith(".docx"):
                text = extract_text_from_docx(content)
            else:
                DOC_FAILURE.inc()
                return {"error": "Unsupported document format."}
            DOC_SUCCESS.inc()
            result = model.infer(text)
            return {"result": result}
        except Exception as e:
            DOC_FAILURE.inc()
            return {"error": str(e)}

    elif input_type == "image":
        if file is None:
            return {"error": "File required for image input."}
        content = await file.read()
        image = load_image(content)
        result = model.infer(image)
        return {"result": result}

    elif input_type == "text":
        if not prompt:
            return {"error": "Prompt required for text input."}
        result = model.infer(prompt)
        return {"result": result}

    else:
        return {"error": "Invalid input_type"}

@app.post("/train/")
async def train(
    model_type: str = Form(...),
    file: UploadFile = File(...)
):
    if model_type not in ["bart", "gpt2"]:
        return {"error": "Training supported only for text models: 'bart' or 'gpt2'"}
    content = await file.read()
    df = pd.read_csv(BytesIO(content))
    if 'source' not in df.columns or 'target' not in df.columns:
        return {"error": "CSV must contain 'source' and 'target' columns"}
    output_dir = f"./finetuned_{model_type}_model"
    fine_tune_text_model(df, MODEL_TYPES[model_type], output_dir)
    # Reload model with fine-tuned weights
    if model_type == "bart":
        models["bart"] = BartModel(output_dir)
    else:
        models["gpt2"] = GPT2Model(output_dir)
    return {"status": f"Fine-tuned {model_type} model saved and loaded."}
