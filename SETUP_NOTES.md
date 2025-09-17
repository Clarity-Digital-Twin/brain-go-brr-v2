# Setup Notes for Seizure Detection Project

## System Requirements

### Tesseract OCR (for PDF extraction)
Some PDFs (like EEG-BIMAMBA.pdf) require OCR to extract text. To enable OCR support:

```bash
# Install Tesseract (requires admin privileges)
sudo apt-get update
sudo apt-get install -y tesseract-ocr tesseract-ocr-eng

# Python packages (already installed)
pip install pytesseract pillow pymupdf[ocr]
```

Once Tesseract is installed, you can convert difficult PDFs with:
```bash
python literature/pdf_to_markdown.py pdfs/EEG-BIMAMBA.pdf --force-ocr
```

## Modern Python Setup (2025)

### UV Package Manager
This project uses `uv` for fast, modern dependency management:

```bash
# Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Setup project
make setup  # or: uv sync --all-extras
```

### Development Workflow

```bash
# Run tests
make test

# Format and lint
make quality

# Train model
make train-local

# See all commands
make help
```

## Literature References

### Key Papers
- **FEMBA**: Main bidirectional Mamba reference for EEG (21,000 hours pre-training)
- **SeizureTransformer**: U-Net + ResCNN architecture reference
- **NEDC/TAES**: Evaluation metrics and scoring

### Note on EEG-BIMAMBA.pdf
This PDF appears to be a scanned document. The extracted images show it contains valuable information about EEG Mamba architectures, but text extraction requires OCR. Install Tesseract (see above) to properly extract the text content.

## Architecture Summary

**Final Stack:** U-Net (1D CNN) → ResCNN stack → Bi-Mamba-2 → U-Net decoder → sigmoid → Hysteresis → TAES

See `FINAL_STACK_ANALYSIS.md` for complete specification.