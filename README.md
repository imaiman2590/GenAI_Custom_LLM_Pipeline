````markdown
# Hugging Face Model Training & Inference API

This project lets you fine-tune and use Hugging Face models for text, documents, or images through a FastAPI backend. It adapts automatically based on the model type and supports monitoring with Prometheus.

---

## Features

- Use any Hugging Face model (like T5, GPT-2, LayoutLMv3, ViT).
- Supports text, PDF/DOCX documents, and image inputs.
- Automatic fine-tuning pipeline based on model type.
- FastAPI endpoints for training and inference.
- Prometheus metrics to track document extraction success/failure.

---

## Project Files

- `main.py` — FastAPI app with `/train` and `/infer` endpoints.
- `models.py` — Loads models dynamically.
- `training_pipeline.py` — Fine-tunes text models.
- `utils.py` — Extracts text from documents, handles images.
- `metrics.py` — Tracks extraction metrics with Prometheus.
- `config.py` — Set model type and name.
- `requirements.txt` — Dependencies.

---

## How to Use

1. Install dependencies:

```bash
pip install -r requirements.txt
````

2. Set your model in `config.py`:

```python
MODEL_TYPE = "text"       # "text", "document", or "image"
MODEL_NAME = "t5-small"   # Hugging Face model name
```

3. Start the server:

```bash
uvicorn main:app --reload
```

---

## API Endpoints

* **POST /train** — Train the model with your dataset (upload CSV or other files).
* **POST /infer** — Run inference on text, documents, or images.
* **GET /metrics** — Prometheus metrics for monitoring.

---

## Docker

You can also run this project with Docker:

Build and run:

```bash
docker build -t hf-finetune-api .
docker run -p 8000:8000 hf-finetune-api
```

Or with Docker Compose:

```bash
docker-compose up --build
```

---

## Supported Models

| Type     | Examples        |
| -------- | --------------- |
| Text     | T5, GPT-2, BART |
| Document | LayoutLMv3      |
| Image    | ViT, DeiT       |

---

## License

MIT License

---

Feel free to ask if you want help with Docker files or example requests!

```

Let me know if you want it even shorter or more detailed!
```
