# Image Background Remover

A polished Flask application for removing the background from uploaded images while preserving image quality.

## Features

- Upload JPG, JPEG, PNG, WEBP, and BMP images up to 30 MB
- Remove backgrounds with transparent PNG output
- Increase an image's JPEG file size from 1 KB up to 30 MB using presets
- Reduce an image's JPEG file size from 1 KB up to 30 MB using presets
- Adjust image ratios using preset ratios, crop-to-fill, or fit-full-image modes
- Download every processed result from the same interface

## Run locally

1. Create a virtual environment
2. Install dependencies
3. Start the Flask app

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/macOS
# or .venv\Scripts\activate  # Windows
pip install -r requirements.txt
python app.py
```

Then open:

```text
http://localhost:5000
```

## Project structure

```text
background_remover/
├── app.py
├── requirements.txt
├── README.md
├── uploads/
├── outputs/
└── templates/
    └── index.html
```
