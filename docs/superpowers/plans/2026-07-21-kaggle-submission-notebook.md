# Kaggle Submission Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Ship `kaggle-submission.ipynb`, a standalone, inference-only notebook that Kaggle can run end-to-end ("Save & Run All") to reproduce `submission.csv` from the already-trained `xlm-roberta-large-xnli` checkpoint, with no retraining.

**Architecture:** Single new notebook containing a stripped-down copy of the inference logic already proven in `watson-notebook.ipynb` cells 106-112 (`NLIModel`, `bert_encode`, batched inference), pointed at Kaggle's standard input/output paths instead of local venv paths. Verified locally first against the existing checkpoint and known-good `submission.csv` using a throwaway scratch copy (not committed), then the real deliverable is written with Kaggle paths and left unexecuted (Kaggle re-executes on "Save & Run All" anyway — a local run would show local paths in stale cell outputs, which is misleading to commit).

**Tech Stack:** PyTorch, Hugging Face Transformers (`AutoTokenizer`/`AutoModel`), pandas — same stack as `watson-notebook.ipynb`, no new dependencies.

## Global Constraints

- Inference-only: no training loop, no augmentation, no candidate comparison. (Spec: "Decisión de alcance")
- Checkpoint (`checkpoints/nli_large_best.pt`, 2.24GB) is not committed and not uploadable via API here (no `kaggle.json`) — the notebook must document, not automate, the manual Kaggle Dataset upload step. (Spec: "Restricción: el checkpoint no está en Kaggle")
- Model: `joeddav/xlm-roberta-large-xnli`, `NLIModel(model_name, dropout=0.4, pooling='mean').float()`, `max_len=128` in `bert_encode`. (Spec: "Contenido del notebook", steps 4-5)
- Output file must be named `submission.csv`, columns `id,prediction`, written under Kaggle's working directory. (Spec: "Contenido del notebook", step 7)
- Do not touch `watson-notebook.ipynb`, `README.md`'s existing results table, or any experiment logic — this is a new, additive file. (Spec: "Fuera de alcance")

---

### Task 1: Verify the inference logic locally before writing the Kaggle deliverable

**Files:**
- Create (scratch, not committed): `C:\Users\alher\.claude\jobs\761a4ed9\tmp\verify_kaggle_inference.py`

**Interfaces:**
- Produces: confirmation that a standalone script using `NLIModel` + `bert_encode` + `joeddav/xlm-roberta-large-xnli` + `checkpoints/nli_large_best.pt` reproduces the existing, already-format-verified `submission.csv` byte-for-byte (same `id` order, same `prediction` values). This de-risks Task 2, which just swaps local paths for Kaggle paths in the same logic — no separate test needed for Task 2 itself.

- [ ] **Step 1: Write the scratch verification script**

Create `C:\Users\alher\.claude\jobs\761a4ed9\tmp\verify_kaggle_inference.py` in the worktree's project root (run from `C:\Users\alher\Desktop\Watson\.claude\worktrees\ethereal-sleeping-rose` so relative paths `checkpoints/` and `test.csv` resolve) with:

```python
import truststore
truststore.inject_into_ssl()

import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

max_len = 128

def bert_encode(premises, hypotheses, tokenizer, max_len=max_len):
    encodings = tokenizer(
        list(premises), list(hypotheses),
        padding='max_length', truncation=True, max_length=max_len,
        return_tensors='pt')
    input_ids = encodings['input_ids']
    if 'token_type_ids' in encodings:
        token_type_ids = encodings['token_type_ids']
    else:
        token_type_ids = torch.zeros_like(input_ids)
    return {
        'input_ids': input_ids,
        'attention_mask': encodings['attention_mask'],
        'token_type_ids': token_type_ids,
    }

class NLIModel(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.0, pooling='cls'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        if self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

model_name = 'joeddav/xlm-roberta-large-xnli'
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = NLIModel(model_name, dropout=0.4, pooling='mean').float().to(device)
model.load_state_dict(torch.load('checkpoints/nli_large_best.pt', map_location=device))
model.eval()

test = pd.read_csv('test.csv')
test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)

batch_size = 32
predictions = []
with torch.no_grad():
    for i in range(0, len(test), batch_size):
        input_ids = test_input['input_ids'][i:i+batch_size].to(device)
        attention_mask = test_input['attention_mask'][i:i+batch_size].to(device)
        token_type_ids = test_input['token_type_ids'][i:i+batch_size].to(device)
        logits = model(input_ids, attention_mask, token_type_ids)
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())

verify_submission = test.id.copy().to_frame()
verify_submission['prediction'] = predictions

existing = pd.read_csv('submission.csv')
merged = existing.merge(verify_submission, on='id', suffixes=('_existing', '_verify'))
mismatches = (merged['prediction_existing'] != merged['prediction_verify']).sum()
print(f'Rows compared: {len(merged)} / existing: {len(existing)} / verify: {len(verify_submission)}')
print(f'Mismatches: {mismatches}')
assert len(merged) == len(existing) == len(verify_submission), 'row count mismatch'
assert mismatches == 0, f'{mismatches} prediction mismatches vs existing submission.csv'
print('OK: inference logic reproduces the existing submission.csv exactly.')
```

This mirrors `watson-notebook.ipynb` cell 112 exactly (same model class, same
`bert_encode`, same checkpoint), so a match here means the same code with
Kaggle paths swapped in (Task 2) will behave identically once run on Kaggle.

- [ ] **Step 2: Run it and confirm it passes**

Run from the worktree root (`C:\Users\alher\Desktop\Watson\.claude\worktrees\ethereal-sleeping-rose`):

```
"C:\Users\alher\Desktop\Watson\.venv\Scripts\python.exe" "C:\Users\alher\.claude\jobs\761a4ed9\tmp\verify_kaggle_inference.py"
```

Expected: `Using device: cuda`, `Mismatches: 0`, and the final `OK:` line. If
`Using device: cpu` prints instead, stop and investigate (per `CLAUDE.md`,
don't treat a CPU fallback run as a valid verification). If mismatches > 0,
stop — it means either the checkpoint path, model config, or encode logic in
this script drifted from what actually produced `submission.csv`; do not
proceed to Task 2 until this passes with 0 mismatches.

No commit for this step — the script is scratch-only, left in the job tmp
directory, not part of the repo.

---

### Task 2: Write the final `kaggle-submission.ipynb` and document its usage

**Files:**
- Create: `kaggle-submission.ipynb` (worktree project root)
- Modify: `CLAUDE.md` (add a usage section)
- Modify: `README.md` (point at the new notebook)

**Interfaces:**
- Consumes: the verified logic from Task 1 (same `NLIModel`, `bert_encode`,
  model name, checkpoint filename), with local paths replaced by Kaggle
  paths and a `CHECKPOINT_PATH` constant that names the expected Kaggle
  Dataset file.
- Produces: nothing consumed by later tasks — this is the last task.

- [ ] **Step 1: Create `kaggle-submission.ipynb`**

Build a notebook (via the `NotebookEdit` tool, or by writing the `.ipynb`
JSON directly) with these cells, in order. Use `nbformat` 4 /
`nbformat_minor` 5, Python 3 kernelspec (`"name": "python3"`, matching what
Kaggle's own kernels register as — do not reuse the local `watson-nli`
kernelspec, which doesn't exist on Kaggle).

**Cell 1 (markdown) — setup instructions:**

```markdown
# Contradictory, My Dear Watson — Kaggle submission

Inference-only notebook. Loads the already-trained `xlm-roberta-large-xnli`
checkpoint (92.99% val_accuracy — see this project's `README.md`/`CLAUDE.md`
for how it was trained) and predicts on the competition's `test.csv`.

**Before running ("Save & Run All"), do these two things in this Kaggle
notebook's settings:**

1. **Add Input → Datasets** → attach a private dataset containing the file
   `nli_large_best.pt` (create it beforehand: Kaggle → Datasets → New
   Dataset → upload `checkpoints/nli_large_best.pt` from this repo, ~2.24GB).
   Update `CHECKPOINT_PATH` below to match the dataset slug Kaggle assigns.
2. **Settings → Internet → On** — needed once, to download the tokenizer
   and base model architecture for `joeddav/xlm-roberta-large-xnli` from
   Hugging Face Hub. The fine-tuned weights themselves come from the
   attached checkpoint, not from the Hub.

Output: `submission.csv`, picked up automatically by Kaggle's
"Submit to Competition" after a successful run.
```

**Cell 2 (code) — paths and imports:**

```python
import pandas as pd
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel

TEST_CSV_PATH = '/kaggle/input/contradictory-my-dear-watson/test.csv'
CHECKPOINT_PATH = '/kaggle/input/watson-nli-large-checkpoint/nli_large_best.pt'
MODEL_NAME = 'joeddav/xlm-roberta-large-xnli'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
```

**Cell 3 (code) — `bert_encode`:**

```python
max_len = 128  # covers 99.4% of premise+hypothesis pairs without truncation

def bert_encode(premises, hypotheses, tokenizer, max_len=max_len):
    encodings = tokenizer(
        list(premises), list(hypotheses),
        padding='max_length', truncation=True, max_length=max_len,
        return_tensors='pt')
    input_ids = encodings['input_ids']
    if 'token_type_ids' in encodings:
        token_type_ids = encodings['token_type_ids']
    else:
        token_type_ids = torch.zeros_like(input_ids)
    return {
        'input_ids': input_ids,
        'attention_mask': encodings['attention_mask'],
        'token_type_ids': token_type_ids,
    }
```

**Cell 4 (code) — `NLIModel`:**

```python
class NLIModel(nn.Module):
    def __init__(self, model_name, num_labels=3, dropout=0.0, pooling='cls'):
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        self.pooling = pooling
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids)
        if self.pooling == 'mean':
            mask = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            summed = torch.sum(outputs.last_hidden_state * mask, dim=1)
            counts = torch.clamp(mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
        else:
            pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)
```

**Cell 5 (code) — load model + checkpoint:**

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = NLIModel(MODEL_NAME, dropout=0.4, pooling='mean').float().to(device)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
model.eval()
```

**Cell 6 (code) — inference + submission.csv:**

```python
test = pd.read_csv(TEST_CSV_PATH)
test_input = bert_encode(test.premise.values, test.hypothesis.values, tokenizer)

batch_size = 32
predictions = []
with torch.no_grad():
    for i in range(0, len(test), batch_size):
        input_ids = test_input['input_ids'][i:i+batch_size].to(device)
        attention_mask = test_input['attention_mask'][i:i+batch_size].to(device)
        token_type_ids = test_input['token_type_ids'][i:i+batch_size].to(device)
        logits = model(input_ids, attention_mask, token_type_ids)
        predictions.extend(logits.argmax(dim=-1).cpu().tolist())

submission = test.id.copy().to_frame()
submission['prediction'] = predictions
submission.to_csv('submission.csv', index=False)
print(f'submission.csv written: {len(submission)} rows')
submission.head()
```

Leave the notebook unexecuted (no cell outputs, `execution_count: null`
throughout) — Kaggle paths don't resolve locally, and Kaggle re-runs
everything on "Save & Run All" regardless of what's saved.

- [ ] **Step 2: Sanity-check the notebook is well-formed**

Run:

```
"C:\Users\alher\Desktop\Watson\.venv\Scripts\python.exe" -c "import json; nb = json.load(open('kaggle-submission.ipynb', encoding='utf-8')); print(len(nb['cells']), 'cells'); print([c['cell_type'] for c in nb['cells']])"
```

Expected: `6 cells` and
`['markdown', 'code', 'code', 'code', 'code', 'code']`. This only checks the
file parses as valid notebook JSON with the right cell shape — it does not
execute the code (Task 1 already verified the logic runs correctly locally).

- [ ] **Step 3: Add a usage section to `CLAUDE.md`**

Add a new top-level section after the existing "## Descargas de Hugging
Face (proxy corporativo)" section (matching the document's existing
Spanish, documentation-heavy style):

```markdown
## Submission a Kaggle (`kaggle-submission.ipynb`)

`watson-notebook.ipynb` genera `submission.csv` localmente, pero esta
competición no acepta subir ese CSV directamente: exige un notebook
ejecutado dentro del entorno de Kaggle ("Save & Run All") cuyo output sea
`submission.csv`, seguido de "Submit to Competition" desde ahí.

`kaggle-submission.ipynb` es ese notebook: separado del notebook de
experimentación, solo hace inferencia con el checkpoint ya entrenado
(`checkpoints/nli_large_best.pt`, el ganador de la comparación de
backbones, 92.99% val_accuracy) — no reentrena nada ni repite la
augmentación por traducción que se usó para entrenarlo, porque esa
augmentación ya está reflejada en los pesos del checkpoint.

Pasos manuales antes de ejecutarlo en Kaggle (no automatizables desde este
repo: no hay `kaggle.json` configurado en esta máquina):

1. Subir `checkpoints/nli_large_best.pt` (2.24GB) como Kaggle Dataset
   privado nuevo, vía la web de Kaggle (Datasets → New Dataset).
2. Adjuntar ese dataset al notebook en Kaggle (Add Input) y actualizar la
   constante `CHECKPOINT_PATH` de la celda 2 con la ruta que Kaggle le
   asigne.
3. Activar Internet en Settings del notebook en Kaggle (necesario para
   descargar el tokenizer y la arquitectura base de
   `joeddav/xlm-roberta-large-xnli` desde Hugging Face Hub).
```

- [ ] **Step 4: Point `README.md` at the new notebook**

In `README.md`, after the line ending in "…el que genera `submission.csv`."
in the "Estado actual" section, add:

```markdown
Para subirlo a Kaggle hace falta `kaggle-submission.ipynb` (esta
competición exige un notebook ejecutado en el propio entorno de Kaggle, no
la subida directa de un CSV) — ver [`CLAUDE.md`](CLAUDE.md) para los pasos
manuales de configuración.
```

- [ ] **Step 5: Commit**

```bash
git add kaggle-submission.ipynb CLAUDE.md README.md
git commit -m "$(cat <<'EOF'
Add standalone Kaggle submission notebook

This competition requires submitting via a notebook executed inside
Kaggle's own environment ("Save & Run All"), not a direct CSV upload.
kaggle-submission.ipynb is inference-only: it loads the already-trained
xlm-roberta-large-xnli checkpoint (92.99% val_accuracy, the winner of
the backbone comparison) and predicts on test.csv, with no retraining
or augmentation -- verified locally to reproduce the existing
submission.csv exactly before writing this Kaggle-path version.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01YSxn7RCahax4pj6LRTPqnZ
EOF
)"
```
