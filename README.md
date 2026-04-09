# Cattle/Buffalo Breed Detector (TFLite)

Streamlit app that loads your exported `breed_model.tflite` and `labels.txt`, then predicts the breed from an uploaded image with confidence (top‑k).

## Setup (Windows / PowerShell)

Create a virtual environment (recommended) and install deps:

```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Add your model files

Create a folder called `models` and place:

- `models/breed_model.tflite`
- `models/labels.txt`

Your `labels.txt` should contain **one label per line**, in the same order used during training/export.

### Recommended (prevents label mismatch)

Also add `models/class_indices.json` (Keras mapping), for example:

```json
{
  "Ayrshire": 0,
  "Brown_Swiss": 1,
  "Gir": 2
}
```

If present, the app will prefer `class_indices.json` to guarantee label order matches the model output indices.

## Run

```bash
streamlit run app.py
```

## Notes

- The app shows the **top prediction + confidence** and a **top‑k table**.
- Use the **confidence threshold** slider to only accept predictions that meet your chosen minimum confidence.

