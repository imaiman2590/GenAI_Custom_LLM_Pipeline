````markdown
🚀 Hugging Face Model Training & Inference API (Multi-Modal)

This project lets you **fine-tune and use Hugging Face models** for **text**, **documents**, or **images** — all through a simple **FastAPI backend**. Whether you're working with plain text, PDFs, Word documents, or images, this pipeline adapts automatically based on the model you're using.

It also includes **Prometheus metrics** for monitoring document extraction performance.

---

 🔧 What This Project Can Do

- **Choose any Hugging Face model at runtime** — like T5, GPT-2, BART, LayoutLMv3, ViT, and more.
- **Supports multiple input types:** text, documents (PDF/DOCX via LayoutLMv3), and images.
- **Fine-tuning pipeline auto-adjusts** based on the selected model.
- **FastAPI endpoints** for uploading data, training models, and making predictions.
- **Built-in Prometheus monitoring** for tracking how well document extraction is working.

---

 🗂 Project Structure

Here's a quick look at the main files:

```bash
├── main.py                # FastAPI app with /train and /infer endpoints
├── models.py              # Loads Hugging Face models dynamically (based on type)
├── training_pipeline.py   # Handles fine-tuning for text models
├── utils.py               # Extracts text from PDFs/DOCX files, handles images
├── metrics.py             # Tracks success/failure of document extraction (Prometheus)
├── config.py              # Set the model type & name here
└── requirements.txt       # Python dependencies
````

---

## 🏁 Getting Started

### 1. Install the dependencies

```bash
pip install -r requirements.txt
```

### 2. Choose your model

Open `config.py` and set the model type and name you'd like to use:

```python
MODEL_TYPE = "text"         # Options: "text", "document", "image"
MODEL_NAME = "t5-small"     # Any model from Hugging Face's hub
```

### 3. Start the FastAPI server

```bash
uvicorn main:app --reload
```

---

## 📡 API Endpoints

### 🧠 `/train` — Train a model on your own dataset

Send a POST request with training data (e.g., CSV) and optional training configs.

**Example JSON body:**

```json
{
  "config": {
    "num_train_epochs": 5,
    "learning_rate": 3e-5
  }
}
```

Include your file as a form upload (`data.csv` with "source" and "target" columns for text tasks).

---

### 🤖 `/infer` — Run inference

Send text, documents, or images, and get back model predictions.

**Example (text):**

```json
{
  "input": "Translate this sentence into French."
}
```

For document or image inputs, use multipart form-data and upload the file.

---

### 📈 `/metrics` — Prometheus metrics

Returns metrics for document extraction:

* `doc_extraction_success_total`
* `doc_extraction_failure_total`

Useful for monitoring and alerting.

---

## ✅ Supported Models

| Model Type | Examples        | Notes                                         |
| ---------- | --------------- | --------------------------------------------- |
| Text       | T5, GPT-2, BART | Fine-tuning handled in `training_pipeline.py` |
| Document   | LayoutLMv3      | For PDF/DOCX layout-based extraction          |
| Image      | ViT, DeiT       | Image inputs supported via `utils.py`         |

---

## 🐳 Docker Support

This project includes a **Dockerfile** and **docker-compose.yaml** for easy setup and deployment.

### Build and Run with Docker

1. **Build the Docker image:**

```bash
docker build -t hf-finetune-api .
```

2. **Run the container:**

```bash
docker run -p 8000:8000 hf-finetune-api
```

This will start the FastAPI server on `http://localhost:8000`.

---

### Using Docker Compose

If you want to run the service with Docker Compose, use:

```bash
docker-compose up --build
```

This will build the image (if needed) and start the container with ports mapped.

---

### API Access

Once running, access the API endpoints as usual:

* Training: `POST http://localhost:8000/train`
* Inference: `POST http://localhost:8000/infer`
* Metrics: `GET http://localhost:8000/metrics`

---

### Why use Docker?

* **Consistency:** Runs the same way regardless of your environment.
* **Isolation:** Keeps dependencies and setups contained.
* **Easy deployment:** Quickly spin up or scale with containers.

---

## 🌱 Next Steps / Ideas

* Add support for fine-tuning image & document models.
* Extend model loading logic to support custom model configs.
* Integrate cloud storage or model registry.
* Add user authentication and access control.

---

## 📜 License

This project is open-source and available under the MIT License.

---

Feel free to open issues or submit PRs for improvements!

```
