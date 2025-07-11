import streamlit as st
import requests
from PIL import Image
import base64
from io import BytesIO

def display_image(b64_img):
    img = Image.open(BytesIO(base64.b64decode(b64_img)))
    st.image(img, caption="Generated Image")

st.title("Multimodal AI App")

input_type = st.selectbox("Input Type", ["text", "image", "document"])
output_type = st.selectbox("Output Type", ["text", "image"] if input_type == "text" else ["text"])

text = ""
file = None

if input_type == "text":
    text = st.text_area("Enter Text")
else:
    file = st.file_uploader("Upload File")

if st.button("Submit"):
    data = {"input_type": input_type, "output_type": output_type}
    files = {"file": (file.name, file.read())} if file else None
    if input_type == "text":
        data["text"] = text

    res = requests.post("http://localhost:8000/infer/", data=data, files=files)
    out = res.json()
    if "error" in out:
        st.error(out["error"])
    else:
        output = out["output"]
        if output_type == "text":
            st.success(output)
        else:
            display_image(output)
