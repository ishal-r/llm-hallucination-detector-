# LLM Hallucination Detector

Detect unsupported, unverifiable, and contradicted claims in AI-generated text — sentence by sentence — using a fine-tuned DeBERTa model trained on the MNLI dataset.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

Given a source document and an AI-generated summary or output, the system analyzes every sentence and labels it as:

- `SUPPORTED` — the claim is entailed by the source (maps from MNLI *entailment*)
- `UNVERIFIABLE` — no relevant evidence found in the source (maps from MNLI *neutral*)
- `CONTRADICTED` — the claim conflicts with the source (maps from MNLI *contradiction*)

It then produces a trust score, a color-coded heatmap, a confusion matrix, and training curves — all from a model you trained yourself.

---

## Project structure

```
llm-hallucination-detector/
│
├── llm_hallucination_detector.ipynb   # main notebook — train, evaluate, serve
├── hallucination_detector.html        # gothic-themed frontend (no framework needed)
├── requirements.txt                   # all Python dependencies
├── .gitignore
└── README.md
```

After running the notebook, these are generated locally (not committed to git):

```
models/hallucination_detector/    # your saved fine-tuned model weights
checkpoints/                      # intermediate training checkpoints
hallucination_heatmap.png         # color-coded sentence verdict heatmap
confusion_matrix.png              # per-class prediction accuracy
training_curves.png               # F1 and loss across epochs
fever_distribution.png            # MNLI label distribution chart
hallucination_results.csv         # full inference results with scores
```

---

## Quickstart

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/llm-hallucination-detector.git
cd llm-hallucination-detector
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. Run the notebook

```bash
jupyter notebook llm_hallucination_detector.ipynb
```

Run cells 1–11 in order. The notebook will:
- Download the MNLI dataset via HuggingFace `datasets`
- Fine-tune `microsoft/deberta-v3-base` on 20,000 training pairs
- Evaluate on 3,000 held-out pairs with F1, precision, recall, and confusion matrix
- Save your trained model to `./models/hallucination_detector`
- Run the inference demo on a built-in Amazon rainforest example

**Training time estimate:**
- GPU (T4 / V100): ~15–25 minutes
- CPU only: ~2–4 hours — set `TRAIN_SAMPLES = 5000` at the top of cell 2 to speed up

### 3. Start the server

Run **cell 12 last**. It starts Flask on `http://localhost:5000`. Keep this cell running.

### 4. Open the web interface
[Open Hallucination Detector](https://ishal-r.github.io/llm-hallucination_detector/hallucination_detector.html)

Open `hallucination_detector.html` in your browser. The header shows **MODEL ONLINE** in green when connected. Paste any source document and AI-generated text, hit **INVOKE JUDGMENT**.

---

## Architecture

```
Source Document
      │
      ▼
Sentence Segmentation (spaCy en_core_web_sm)
      │
      ▼
Semantic Evidence Retrieval
(all-MiniLM-L6-v2 cosine similarity — top-k source sentences per claim)
      │
      ▼
Fine-tuned DeBERTa-v3-base NLI Classifier
(trained on MNLI: entailment / neutral / contradiction)
      │
      ▼
Label Aggregation (average logits across top-k evidence pairs)
      │
      ▼
Trust Score + Verdicts
      │
      ├── Color-coded heatmap (matplotlib)
      ├── CSV export
      └── Flask API → Gothic HTML frontend
```

---

## Training details

| Setting | Value |
|---|---|
| Base model | `microsoft/deberta-v3-base` |
| Dataset | MNLI via HuggingFace `glue/mnli` |
| Task | 3-class NLI classification |
| Label mapping | entailment → SUPPORTED, neutral → UNVERIFIABLE, contradiction → CONTRADICTED |
| Train samples | 20,000 (configurable via `TRAIN_SAMPLES`) |
| Eval split | `validation_matched` — 3,000 samples |
| Epochs | 3 |
| Batch size | 8 |
| Learning rate | 2e-5 with warmup |
| Max sequence length | 256 tokens |
| Optimizer | AdamW + weight decay 0.01 |
| Mixed precision | fp16 when GPU available |

---

## Dataset — MNLI

[Multi-Genre NLI (MNLI)](https://huggingface.co/datasets/nyu-mll/multi_nli) contains ~393k sentence pairs from a diverse range of genres (fiction, government, telephone conversations, travel guides, etc.), labeled by crowd workers as:

| MNLI label | Our label | Meaning |
|---|---|---|
| entailment | SUPPORTED | Premise logically implies the hypothesis |
| neutral | UNVERIFIABLE | Premise and hypothesis are unrelated |
| contradiction | CONTRADICTED | Premise and hypothesis cannot both be true |

The dataset is perfectly balanced (~130k per class) and loads automatically via `datasets.load_dataset("glue", "mnli")` — no manual download needed.

---

## Outputs

| File | Description |
|---|---|
| `confusion_matrix.png` | Per-class prediction accuracy heatmap |
| `training_curves.png` | Macro F1 and eval loss across training epochs |
| `fever_distribution.png` | MNLI label balance in the training subset |
| `hallucination_heatmap.png` | Color-coded sentence-level verdicts |
| `hallucination_results.csv` | Full results — labels, confidence scores, evidence |

---

## Web interface

The `hallucination_detector.html` file is a single-page app with a gothic pixel-art aesthetic. It requires no build step — just open it in a browser while the Flask server (cell 12) is running.

The header shows a live server status indicator:
- **MODEL ONLINE** (green) — Flask server is running, ready to analyze
- **SERVER OFFLINE** (red) — run cell 12 first

---

## Use cases

- Auditing RAG pipeline outputs for factual grounding
- Trust and safety review of LLM-generated summaries
- QA pipelines for AI writing assistants
- Detecting hallucination in automated report generation

---

## Limitations

- MNLI covers general-domain text — performance may degrade on highly technical domains (medical, legal) without further fine-tuning
- Evidence retrieval is embedding-based (top-k cosine similarity), not exhaustive — may miss relevant passages in very long source documents
- Works best on factual and informational prose

---

## License

MIT. Free to use, modify, and distribute.

---

## Roadmap

- [ ] Fine-tune on domain-specific NLI data (MedNLI for healthcare, CaseHold for legal)
- [ ] PDF source document support via PyMuPDF
- [ ] Batch evaluation across multiple AI outputs
- [ ] Publish fine-tuned model to HuggingFace Hub
- [ ] Streamlit interface as alternative frontend
