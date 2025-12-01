# 🚀 Transformer English → Hindi Translation (From Scratch in PyTorch)

This repository contains a complete **Transformer Encoder–Decoder architecture implemented entirely from scratch** using PyTorch.  
Inspired by **“Attention Is All You Need” (Vaswani et al., 2017)**, the project manually implements every component of the Transformer without using `torch.nn.Transformer`.

It is designed for **education, research, and professional ML/NLP portfolio demonstration**.

---

# 🌐 Pretrained Model (HuggingFace Hub)

The trained model weights (`transformer_model.pth`) are hosted on Hugging Face:

👉 **[Click here to download from HuggingFace](https://huggingface.co/ankpatil1203/transformer-eng-hin-from-scratch/blob/main/transformer_model.pth)**

### Load the model in Python:

```python
from huggingface_hub import hf_hub_download
import torch
from src.model import Transformer

# Download model from HuggingFace Hub
model_path = hf_hub_download(
    repo_id="ankpatil1203/transformer-eng-hin-from-scratch",
    filename="transformer_model.pth"
)

# Initialize architecture and load weights
model = Transformer(src_vocab_size, trg_vocab_size)
model.load_state_dict(torch.load(model_path, map_location="cpu"))
model.eval()

```

---

# 📘 Project Links (GitHub Navigation)

| Component | Link |
|----------|------|
| 📓 Jupyter Notebook | [Click here](https://github.com/Ankush-Patil99/transformer-seq2seq-from-scratch/blob/main/transformer-seq2seq-from-scratch/notebooks/transformers-eng-hin.ipynb) |
| 📊 Results (plots, metrics, translations) | [Click here](https://github.com/Ankush-Patil99/transformer-seq2seq-from-scratch/tree/main/transformer-seq2seq-from-scratch/results) |
| 🔤 Vocabulary Files | [Click here](https://github.com/Ankush-Patil99/transformer-seq2seq-from-scratch/tree/main/transformer-seq2seq-from-scratch/vocab) |
| 🔗 Dataset (Kaggle) | [Click here](https://www.kaggle.com/datasets/vaibhavkumar11/hindi-english-parallel-corpus) |

---

# 📚 Dataset

This project uses the **Hindi–English Parallel Corpus** from Kaggle.

Dataset link:  
👉 **[Click here to view/download the dataset](https://www.kaggle.com/datasets/vaibhavkumar11/hindi-english-parallel-corpus)**

### ⚠ Important  
The dataset **is not included** in this repository due to size and license restrictions.

To train the model:

1. Download the dataset from Kaggle  
2. Place the file into:

```
data/raw/
```

3. `train.py` will automatically split into training & validation sets.

---

# 🧠 Model Architecture

The Transformer follows the Encoder–Decoder architecture with:

- Multi-Head Self-Attention  
- Cross-Attention  
- Feed-Forward Networks  
- Positional Encoding  
- Layer Normalization  
- Skip Connections  
- Look-Ahead Masking  

### Diagram

```
Input → Token Embedding → Positional Encoding → Encoder (N layers)
                                                        ↓
                       Target → Embedding → Positional Encoding → Decoder (N layers)
                                                        ↓
                                            Linear → Softmax → Output Tokens
```

### Hyperparameters

| Component | Value |
|----------|--------|
| d_model | 256 |
| heads | 4 |
| encoder layers | 3 |
| decoder layers | 3 |
| FFN size | 512 |
| dropout | 0.1 |
| optimizer | Adam (lr = 3e-4) |
| loss | Label Smoothing Loss |
| decoding | Greedy & Beam Search |

---

# ⚙️ Installation

```bash
pip install torch einops sacrebleu matplotlib pandas huggingface_hub
```

---

# 🏋️ Training Instructions

Run training:

```bash
python src/train.py
```

The training script performs:

- Data loading  
- Cleaning & preprocessing  
- Tokenization  
- Vocabulary building  
- Train/validation split  
- Mask creation  
- Forward/backward passes  
- Label smoothing  
- BLEU evaluation  
- Loss tracking  
- Model saving  

---

## 🧪 Inference After Loading Model

After calling `model.eval()`, you can translate English sentences to Hindi:

```python
import torch, json
from src.utils import preprocess_sentence, decode_ids, create_padding_mask
from src.model import beam_search

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# Load vocabularies
with open("vocab/vocab_en.json") as f:
    vocab_en = json.load(f)

with open("vocab/vocab_hi.json") as f:
    vocab_hi = json.load(f)

sentence = "give your application a workout"

# Convert input sentence → token IDs
src_ids = preprocess_sentence(sentence, vocab_en)
src_tensor = torch.tensor([src_ids]).to(device)

# Build mask
src_mask = create_padding_mask(src_tensor, pad_id=0).to(device)

# Beam search decoding
pred_ids = beam_search(model, src_tensor, src_mask, beam_width=5)

# Convert token IDs → Hindi text
translation = decode_ids(pred_ids.tolist(), vocab_hi)

print("Input      :", sentence)
print("Translation:", translation)

```

---

# 🎯 Compute BLEU Score

```python
from src.utils import compute_bleu

bleu = compute_bleu(model, val_loader, vocab_en, vocab_hi)
print("BLEU Score:", bleu)
```

---

# 🔥 Visualizations

### 📉 Training Loss Curve  
Saved at:

```
results/images/loss_curve.png
```

### 🎯 Attention Heatmap  
Saved at:

```
results/images/attention_heatmap.png
```

---

# 📊 Results Summary

| Metric | Value |
|--------|--------|
| BLEU Score | XX.XX (update with your score) |

---

# 📝 Sample Translations

A file containing sample predictions is available at:

```
results/metrics/sample_translations.csv
```

---

# 🔥 Why This Project Matters

This project demonstrates:

- Strong understanding of Transformer internals  
- Ability to implement NLP architectures manually  
- Clean engineering practices  
- Multi-step training & evaluation workflow  
- Visualization and analysis skills  
- Integration with HuggingFace Hub  
- Portfolio-ready structure and documentation  

Perfect for ML Engineer / NLP Engineer roles.

---

# 🔮 Future Work Suggestions

- Add LSTM / GRU Seq2Seq baseline for comparison  
- Add Gradio UI for real-time translation  
- Train a deeper Transformer  
- Add FastAPI-based deployment  
- Use pretrained embeddings (FastText, GloVe)  
- Add mixed-precision training  

---

# 📄 License

MIT License

---

# 👨‍💻 Author

**Ankush Patil**  
Machine Learning & NLP Engineer  
Deep Learning | Transformers | PyTorch

---

# ⭐ Support

If you found this project helpful, please consider giving it a **GitHub Star** ⭐
