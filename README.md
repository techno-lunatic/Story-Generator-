# Novel Generator

A small Streamlit interface for inference with the custom PyTorch language model from `NovelGenerator.ipynb`. The repository loads a trained checkpoint and its matching BPE tokenizer; it does not train a model.

## Repository layout

```text
NovelGenerator/
├── app.py
├── inference.py
├── model.py
├── data/
│   └── input/                  # Place the source PDF here for reproducible preparation
├── artifacts/
│   └── final_model/            # Reusable model checkpoint and matching tokenizer
├── .gitignore
├── README.md
├── notebooks
|    └── GPT_From_Scratch.ipynb
|    └── NovelGenerator.ipynb    # Raw notebook of the project
└── requirements.txt
```

## Model files

Put the two compatible files below in `artifacts/final_model/`:

- `best_model.pth` — checkpoint created by the notebook's best-model save step.
- `tokenizer.json` — the ByteLevel BPE tokenizer used to train that checkpoint.

The notebook checkpoint does not include the tokenizer, so retain the tokenizer saved from the same training run. The model configuration in `inference.py` matches the notebook values: vocabulary size is read from the tokenizer; embedding dimension 256, 4 attention heads, 4 transformer layers, context length 128, and dropout 0.1.

The `data/input/` directory is where the source PDF belongs when repeating the notebook's PDF extraction and tokenizer/model preparation. Keep the same source document and tokenizer-training choices for reproducibility. PDF processing and model training are outside this inference-only repository; they are not run when starting the app. Do not commit the PDF or model files by default: `.gitignore` excludes them.

## Run locally

Use Python 3.10 or newer. Install the listed dependencies in a virtual environment, then start Streamlit from this directory:

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS/Linux
# source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

The interface provides a prompt box and controls for `top_k`, temperature, and `max_new_tokens`. Generation uses CUDA when available and otherwise uses the CPU. The app reports a clear message if the checkpoint or tokenizer files are missing.

## Deployment

Deploy the repository with its dependencies and provide `best_model.pth` and `tokenizer.json` under `artifacts/final_model/` in the deployment environment. Large model files may need to be stored with your deployment provider's artifact or model storage rather than committed to Git. Confirm the provider supports the required PyTorch version and the model's memory requirements.
