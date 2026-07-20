# Mejora de accuracy (Kaggle) — submission fix + experimentos de backbone Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Corregir `submission.csv` (usa el modelo baseline en vez del mejor modelo entrenado) y evaluar si un backbone distinto (mismo tamaño con más pretraining NLI, o genuinamente más grande) mejora sobre `model_nli_aug` (88.41% val_accuracy), el mejor modelo actual.

**Architecture:** Todo el trabajo vive en `watson-notebook.ipynb` (100 celdas, índices 0–99 al momento de escribir este plan) como nuevas celdas al final, siguiendo el patrón ya establecido de cada sección (`model_nli`, `model_nli_aug`, `model_nli_dapt`, `model_nli_gradual`): no se sustituyen celdas anteriores, solo se añaden nuevas. Una única excepción: la celda 23 (`bert_encode`) se edita para tolerar tokenizers que no devuelven `token_type_ids` (necesario para el backbone XLM-R de la Tarea 3), de forma retrocompatible.

**Tech Stack:** PyTorch + Hugging Face Transformers, mismo venv `.venv` (`watson-nli` kernel) ya configurado. Dos checkpoints nuevos vía Hugging Face Hub: `MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (Tarea 2) y `joeddav/xlm-roberta-large-xnli` (Tarea 3) — ambos ya verificados como descargables y compatibles con `AutoModel`/`AutoTokenizer` en este entorno (proxy con `truststore`).

## Global Constraints

- SEED = 42 (celda 4), fijado antes de cualquier instanciación de modelo — no lo toques.
- `max_len = 128` (celda 23) es constante compartida — no cambiarla.
- Reutilizar siempre el mismo split `train_idx`/`val_idx` (celda 25, `test_size=0.2, random_state=42`) y el mismo `val_loader_nli` (celda 47) para que las comparaciones entre modelos sean justas.
- `checkpoint_dir = 'checkpoints'` (no versionado, ver `.gitignore`) — todos los checkpoints nuevos van ahí con el mismo patrón `{nombre}_last.pt` / `{nombre}_best.pt`.
- `class_names = ['entailment', 'neutral', 'contradiction']` — orden de labels fijo en todo el notebook, no reordenar.
- GPU: RTX 4060 Laptop, 8GB VRAM. La Tarea 3 requiere que `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay` ya se haya subido y la máquina reiniciado — es un cambio de registro de Windows que requiere confirmación explícita del usuario en el momento, nunca asumir que ya está hecho.
- Ejecución del notebook: correr las celdas 0–66 en orden (contienen toda la preparación de datos, el baseline, `model_nli` y `model_nli_aug` — necesarias como prerequisito de las Tareas 1–4) y **saltarse las celdas 67–99** (ensemble, DAPT, verificación de leakage, checkpoint averaging, gradual unfreezing) — ninguna de las tareas de este plan depende de ellas. Como optimización opcional (los checkpoints `baseline_best.pt`, `nli_best.pt`, `nli_aug_best.pt` ya existen en disco de una ejecución anterior), se pueden saltar los bucles de entrenamiento de las celdas 30/49/62 y en su lugar cargar el `state_dict` guardado directamente — en ese caso, fijar a mano `baseline_best_val_accuracy`/`nli_best_val_accuracy`/`nli_aug_best_val_accuracy` con los valores ya documentados (65.9%/88.0%/88.41%) para que las celdas posteriores que referencian esas variables no fallen con `NameError`.

---

## Task 1: Fix de `submission.csv` (P0) — usar `model_nli_aug`, no el baseline

**Files:**
- Modify: `watson-notebook.ipynb` — añadir 2 celdas nuevas al final (tras la celda 99: 1 markdown + 1 código).

**Interfaces:**
- Consumes: `model_nli_aug` (entrenado y con `nli_aug_best.pt` cargado, celda 64), `tokenizer_nli`, `model_name_nli`, `nli_aug_best_val_accuracy`, `bert_encode`, `batch_size`, `checkpoint_dir`, `device`, `NLIModel` — todos ya presentes tras ejecutar las celdas 0–66.
- Produces: `submission.csv` regenerado en disco; variable `candidates` (dict) en el notebook, que la Tarea 4 amplía.

- [ ] **Step 1: Ejecutar las celdas 0–66 del notebook** (kernel `watson-nli`), para reconstruir el estado necesario (`model_nli_aug` entrenado, `tokenizer_nli`, `nli_aug_best_val_accuracy`, etc.). Ver nota de optimización en "Global Constraints" si quieres evitar reentrenar baseline/`model_nli`/`model_nli_aug` desde cero.

- [ ] **Step 2: Añadir celda markdown al final del notebook**

```markdown
## Submission final: regenerar `submission.csv` con el mejor modelo entrenado

`submission.csv` se generaba hasta ahora con las celdas 33–40, que usan `model`
(el baseline `bert-base-multilingual-cased`, ~65.9% val_accuracy) — ninguna
celda de las secciones de transfer learning con mDeBERTa vuelve a generar
predicciones sobre `test.csv`. Esta celda corrige eso: genera la submission
con el mejor modelo disponible (por ahora, `model_nli_aug`, 88.41%
val_accuracy). La celda queda escrita para poder ampliarse con más
candidatos sin reescribirla — ver la sección "elegir el mejor modelo" más
abajo, que la reutiliza.
```

- [ ] **Step 3: Añadir celda de código al final del notebook**

```python
candidates = {
    'model_nli_aug': (os.path.join(checkpoint_dir, 'nli_aug_best.pt'), model_name_nli,
                       tokenizer_nli, nli_aug_best_val_accuracy),
}

best_name = max(candidates, key=lambda name: candidates[name][3])
best_checkpoint, best_model_name, best_tokenizer, best_accuracy = candidates[best_name]
print(f'Mejor modelo: {best_name} (val_accuracy={best_accuracy:.4f})')

best_model = NLIModel(best_model_name, dropout=0.4, pooling='mean').float().to(device)
best_model.load_state_dict(torch.load(best_checkpoint))
best_model.eval()

test = pd.read_csv('test.csv')
test_input_final = bert_encode(test.premise.values, test.hypothesis.values, best_tokenizer)

predictions_final = []
with torch.no_grad():
    for i in range(0, len(test), batch_size):
        input_ids = test_input_final['input_ids'][i:i+batch_size].to(device)
        attention_mask = test_input_final['attention_mask'][i:i+batch_size].to(device)
        token_type_ids = test_input_final['token_type_ids'][i:i+batch_size].to(device)
        logits = best_model(input_ids, attention_mask, token_type_ids)
        predictions_final.extend(logits.argmax(dim=-1).cpu().tolist())

submission = test.id.copy().to_frame()
submission['prediction'] = predictions_final
submission.to_csv('submission.csv', index=False)
print(f'submission.csv regenerado con {best_name} (val_accuracy={best_accuracy:.4f}), {len(submission)} filas')
```

- [ ] **Step 4: Ejecutar la celda y verificar la salida**

Expected: se imprime `Mejor modelo: model_nli_aug (val_accuracy=0.8841)` (el valor exacto de `nli_aug_best_val_accuracy` puede variar unas décimas por no-determinismo de cuDNN) y `submission.csv regenerado con model_nli_aug (val_accuracy=0.8841), 5195 filas`. Verificar además con `pd.read_csv('submission.csv').shape` que da `(5195, 2)` y que `prediction` solo contiene `{0, 1, 2}`.

- [ ] **Step 5: Guardar el notebook y commitear**

```bash
git add watson-notebook.ipynb
git commit -m "$(cat <<'EOF'
Fix submission.csv to use the best trained model instead of the baseline

submission.csv was generated from the untouched-for-NLI mBERT baseline
(~65.9% val_accuracy); no cell after the mDeBERTa transfer-learning
section ever regenerated test predictions. This adds a cell that loads
model_nli_aug (88.41% val_accuracy, the best checkpoint so far) and
regenerates submission.csv from it.
EOF
)"
```

---

## Task 2: Vía A — swap de checkpoint pretraining (P1) — `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`

**Files:**
- Modify: `watson-notebook.ipynb` — añadir 4 celdas nuevas al final (1 markdown + 3 código), después de las de la Tarea 1.

**Interfaces:**
- Consumes: `tokenizer_nli` (verificado idéntico — mismo vocab, mismos special tokens, mismo `input_ids` de salida — al tokenizer de `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, así que se reutiliza sin re-tokenizar), `train_loader_nli_aug`, `val_loader_nli`, `val_lang_nli`, `lang_acc_aug`, `K`, `criterion`, `checkpoint_dir`, `device`, `NLIModel`, `get_linear_schedule_with_warmup`.
- Produces: `model_nli_pretrain`, `nli_pretrain_best_val_accuracy`, `lang_acc_pretrain`, checkpoints `nli_pretrain_last.pt`/`nli_pretrain_best.pt` — consumidos por la Tarea 4.

- [ ] **Step 1: Añadir celda markdown**

```markdown
## Mejora: swap de checkpoint pretraining (mismo tamaño) — `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`

El backbone actual (`mDeBERTa-v3-base-mnli-xnli`) y este checkpoint son el
mismo modelo base (mDeBERTa-v3-base, mismo tokenizer — verificado que
producen exactamente el mismo `input_ids` para el mismo texto), pero este
se entrenó sobre más pares NLI multilingües (~2.7M vs el MNLI+XNLI
original). Reutilizamos `tokenizer_nli`, `train_loader_nli_aug` y
`val_loader_nli` tal cual — la única variable que cambia frente a
`model_nli_aug` es de qué checkpoint parte el backbone.
```

- [ ] **Step 2: Añadir celda de código — modelo y freezing**

```python
model_name_nli_pretrain = 'MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7'

model_nli_pretrain = NLIModel(model_name_nli_pretrain, dropout=0.4, pooling='mean').float().to(device)

for name, param in model_nli_pretrain.bert.named_parameters():
    if name.startswith('encoder.layer.'):
        layer_idx = int(name.split('.')[2])
        param.requires_grad = layer_idx >= (model_nli_pretrain.bert.config.num_hidden_layers - K)
    else:
        param.requires_grad = False

for param in model_nli_pretrain.classifier.parameters():
    param.requires_grad = True

trainable_params_nli_pretrain = sum(p.numel() for p in model_nli_pretrain.parameters() if p.requires_grad)
print(f'Parametros entrenables: {trainable_params_nli_pretrain:,}')
```

Expected al ejecutar: descarga el checkpoint (ya cacheado si se corrió la verificación previa) e imprime `Parametros entrenables: <mismo número que trainable_params_nli_aug>` (mismo `K=3`, mismo tamaño de modelo).

- [ ] **Step 3: Añadir celda de código — entrenamiento con early stopping**

```python
optimizer_nli_pretrain = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model_nli_pretrain.parameters()), lr=2e-5, weight_decay=0.1)

epochs_nli_pretrain = 5
patience_nli_pretrain = 3
total_steps_nli_pretrain = len(train_loader_nli_aug) * epochs_nli_pretrain
scheduler_nli_pretrain = get_linear_schedule_with_warmup(
    optimizer_nli_pretrain,
    num_warmup_steps=int(0.1 * total_steps_nli_pretrain),
    num_training_steps=total_steps_nli_pretrain)

best_val_loss_nli_pretrain = float('inf')
nli_pretrain_best_val_accuracy = None
epochs_no_improve_nli_pretrain = 0

for epoch in range(epochs_nli_pretrain):
    model_nli_pretrain.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    for batch in train_loader_nli_aug:
        optimizer_nli_pretrain.zero_grad()
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        batch_labels = batch['label'].to(device)

        logits = model_nli_pretrain(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, batch_labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model_nli_pretrain.parameters(), max_norm=1.0)
        optimizer_nli_pretrain.step()
        scheduler_nli_pretrain.step()

        train_loss += loss.item() * batch_labels.size(0)
        train_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
        train_total += batch_labels.size(0)

    model_nli_pretrain.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader_nli:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            batch_labels = batch['label'].to(device)

            logits = model_nli_pretrain(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, batch_labels)

            val_loss += loss.item() * batch_labels.size(0)
            val_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
            val_total += batch_labels.size(0)

    nli_pretrain_val_accuracy = val_correct / val_total
    avg_val_loss_nli_pretrain = val_loss / val_total
    print(f'Epoch {epoch+1}/{epochs_nli_pretrain} - '
          f'loss: {train_loss/train_total:.4f} - accuracy: {train_correct/train_total:.4f} - '
          f'val_loss: {avg_val_loss_nli_pretrain:.4f} - val_accuracy: {nli_pretrain_val_accuracy:.4f}')

    torch.save(model_nli_pretrain.state_dict(), os.path.join(checkpoint_dir, 'nli_pretrain_last.pt'))

    if avg_val_loss_nli_pretrain < best_val_loss_nli_pretrain:
        best_val_loss_nli_pretrain = avg_val_loss_nli_pretrain
        nli_pretrain_best_val_accuracy = nli_pretrain_val_accuracy
        epochs_no_improve_nli_pretrain = 0
        torch.save(model_nli_pretrain.state_dict(), os.path.join(checkpoint_dir, 'nli_pretrain_best.pt'))
        print(f'  -> nuevo mejor val_loss ({best_val_loss_nli_pretrain:.4f}), guardado en nli_pretrain_best.pt')
    else:
        epochs_no_improve_nli_pretrain += 1
        print(f'  -> val_loss sin mejora desde hace {epochs_no_improve_nli_pretrain} epoch(s) (mejor: {best_val_loss_nli_pretrain:.4f})')
        if epochs_no_improve_nli_pretrain >= patience_nli_pretrain:
            print(f'Early stopping: sin mejora en val_loss durante {patience_nli_pretrain} epochs consecutivos.')
            break
```

Expected: imprime una línea `Epoch N/5 - ...` por época (hasta 5, o menos si el early stopping corta antes) y al menos un `-> nuevo mejor val_loss (...)`.

- [ ] **Step 4: Añadir celda de código — recargar mejor checkpoint, evaluar y comparar**

```python
model_nli_pretrain.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'nli_pretrain_best.pt')))
model_nli_pretrain.eval()

val_preds_nli_pretrain, val_true_nli_pretrain = [], []
with torch.no_grad():
    for batch in val_loader_nli:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        batch_labels = batch['label'].to(device)

        logits = model_nli_pretrain(input_ids, attention_mask, token_type_ids)
        val_preds_nli_pretrain.extend(logits.argmax(dim=-1).cpu().tolist())
        val_true_nli_pretrain.extend(batch_labels.cpu().tolist())

lang_df_pretrain = pd.DataFrame({'language': val_lang_nli, 'true': val_true_nli_pretrain, 'pred': val_preds_nli_pretrain})
lang_df_pretrain['correct'] = lang_df_pretrain['true'] == lang_df_pretrain['pred']
lang_acc_pretrain = lang_df_pretrain.groupby('language')['correct'].agg(accuracy='mean', n='size')

comparison_pretrain = pd.DataFrame({
    'mnli-xnli (model_nli_aug)': lang_acc_aug['accuracy'],
    '2mil7-pretrain (model_nli_pretrain)': lang_acc_pretrain['accuracy'],
})
comparison_pretrain.loc['GLOBAL'] = [nli_aug_best_val_accuracy, nli_pretrain_best_val_accuracy]
comparison_pretrain['delta'] = (comparison_pretrain['2mil7-pretrain (model_nli_pretrain)']
                                 - comparison_pretrain['mnli-xnli (model_nli_aug)'])
comparison_pretrain.sort_values('delta', ascending=False)
```

Expected: una tabla con una fila por idioma + `GLOBAL`, columna `delta` mostrando la diferencia frente a `model_nli_aug`. Verificar que la fila `GLOBAL` de `nli_pretrain_best_val_accuracy` esté en un rango plausible (0.80–0.92, consistente con el resto de experimentos del notebook) — un valor fuera de ese rango indica un bug (p.ej. tokenizer o loader mal conectado), no una mejora real.

- [ ] **Step 5: Guardar el notebook y commitear**

```bash
git add watson-notebook.ipynb
git commit -m "$(cat <<'EOF'
Add mDeBERTa checkpoint-swap experiment (2mil7 pretrain vs mnli-xnli)

Tests whether more NLI pretraining data on the same-size backbone beats
model_nli_aug, isolating pretraining checkpoint as the only variable
(same tokenizer, same K=3 freezing, same augmented training data).
EOF
)"
```

---

## Task 3: Vía B — backbone genuinamente más grande (P2) — `xlm-roberta-large-xnli`

**Precondition (manual, no automatizable):** confirmar explícitamente con el usuario que `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay` ya se subió (p.ej. a 60s) y la máquina se reinició, antes de ejecutar cualquier celda de entrenamiento de esta tarea. No asumir que ya está hecho solo porque se acordó durante el brainstorming.

**Files:**
- Modify: `watson-notebook.ipynb:cell 23` (`bert_encode`) — fix retrocompatible para tokenizers sin `token_type_ids`.
- Modify: `watson-notebook.ipynb` — añadir 5 celdas nuevas al final (1 markdown + 4 código), después de las de la Tarea 2.

**Interfaces:**
- Consumes: `bert_encode` (editada), `train_aug_df`, `train_labels_aug`, `labels`, `val_idx`, `train`, `NLIDataset`, `K`, `criterion`, `checkpoint_dir`, `device`, `NLIModel`, `get_linear_schedule_with_warmup`, `val_lang_nli`, `lang_acc_aug`, `lang_acc_pretrain`, `nli_aug_best_val_accuracy`, `nli_pretrain_best_val_accuracy`.
- Produces: `model_nli_large`, `tokenizer_nli_large`, `model_name_nli_large`, `nli_large_best_val_accuracy`, checkpoints `nli_large_last.pt`/`nli_large_best.pt` — consumidos por la Tarea 4.

- [ ] **Step 1: Confirmar con el usuario que el precondition de arriba (TdrDelay + reinicio) está cumplido.** No continuar sin confirmación explícita.

- [ ] **Step 2: Editar la celda 23 (`bert_encode`) para tolerar tokenizers sin `token_type_ids`**

Verificado: el tokenizer de `joeddav/xlm-roberta-large-xnli` (estilo RoBERTa/XLM-R) no devuelve `token_type_ids` — la llamada actual a `encodings['token_type_ids']` daría `KeyError`. Los tokenizers ya usados (mBERT, mDeBERTa) sí lo devuelven, así que este fix no cambia su comportamiento.

Reemplazar el contenido completo de la celda 23 por:

```python
max_len = 128  # con el tokenizer de mDeBERTa, max_len=50 truncaba el 45.8% de los ejemplos (mediana premise+hypothesis: 48 tokens); 128 cubre el 99.4%

def bert_encode(premises, hypotheses, tokenizer, max_len=max_len):
    encodings = tokenizer(
        list(premises), list(hypotheses),
        padding='max_length', truncation=True, max_length=max_len,
        return_tensors='pt')
    input_ids = encodings['input_ids']
    # Los tokenizers estilo RoBERTa/XLM-R (p.ej. xlm-roberta-large-xnli) no devuelven
    # token_type_ids -- rellenamos con ceros para no tener que distinguir arquitecturas
    # en el resto del pipeline (NLIModel.forward, los bucles de entrenamiento).
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

Re-ejecutar esta celda (y todo lo que dependa de ella aguas abajo, si el kernel ya había corrido `bert_encode` antes con la versión vieja) antes de continuar.

- [ ] **Step 3: Añadir celda markdown**

```markdown
## Mejora: backbone genuinamente más grande — `xlm-roberta-large-xnli`

`joeddav/xlm-roberta-large-xnli` (XLM-RoBERTa-large fine-tuneado en XNLI,
~560M parámetros, 24 capas — ~2x mDeBERTa-base) es un tokenizer distinto
(vocabulario XLM-R, sin `token_type_ids`), así que retokenizamos
`train_aug_df` y el split de validación. Reducimos `batch_size` a la mitad
y usamos gradient accumulation (batch efectivo = 32, igual que el resto)
para no sobrecargar los 8GB de VRAM incluso con el TdrDelay ya subido.
`K=3` capas superiores entrenables (de 24) — misma fracción pequeña que en
las demás secciones; la lección de gradual unfreezing es que más capacidad
entrenable no mueve el techo, así que no la ampliamos solo por tener más
capas disponibles.
```

- [ ] **Step 4: Añadir celda de código — liberar memoria, tokenizer y loaders nuevos**

```python
import gc
if 'model_nli_pretrain' in globals():
    del model_nli_pretrain
gc.collect()
torch.cuda.empty_cache()

model_name_nli_large = 'joeddav/xlm-roberta-large-xnli'
tokenizer_nli_large = AutoTokenizer.from_pretrained(model_name_nli_large)

train_input_nli_large = bert_encode(train_aug_df.premise.values, train_aug_df.hypothesis.values, tokenizer_nli_large)
train_dataset_nli_large = NLIDataset(train_input_nli_large, train_labels_aug)

val_input_nli_large = bert_encode(train.iloc[val_idx].premise.values, train.iloc[val_idx].hypothesis.values, tokenizer_nli_large)
val_dataset_nli_large = NLIDataset(val_input_nli_large, labels[val_idx])

batch_size_large = 16  # mitad de batch_size=32: xlm-roberta-large tiene ~2x parametros y
                        # ~2x capas (24 vs 12) que mDeBERTa-base, mas activaciones en VRAM
grad_accum_steps_large = 2  # batch efectivo = 16*2 = 32, igual que el resto de secciones

train_loader_nli_large = DataLoader(train_dataset_nli_large, batch_size=batch_size_large, shuffle=True)
val_loader_nli_large = DataLoader(val_dataset_nli_large, batch_size=batch_size_large)
```

Expected: sin error de `KeyError: 'token_type_ids'` (confirma que el fix de la celda 23 funciona).

- [ ] **Step 5: Añadir celda de código — modelo y freezing**

```python
model_nli_large = NLIModel(model_name_nli_large, dropout=0.4, pooling='mean').float().to(device)

K_large = 3  # capas superiores entrenables de 24 -- misma fraccion pequeña que K=3/12 en las
             # demas secciones (ver celda markdown de arriba)

for name, param in model_nli_large.bert.named_parameters():
    if name.startswith('encoder.layer.'):
        layer_idx = int(name.split('.')[2])
        param.requires_grad = layer_idx >= (model_nli_large.bert.config.num_hidden_layers - K_large)
    else:
        param.requires_grad = False

for param in model_nli_large.classifier.parameters():
    param.requires_grad = True

trainable_params_nli_large = sum(p.numel() for p in model_nli_large.parameters() if p.requires_grad)
total_params_nli_large = sum(p.numel() for p in model_nli_large.parameters())
print(f'Parametros entrenables: {trainable_params_nli_large:,} de {total_params_nli_large:,} '
      f'({trainable_params_nli_large/total_params_nli_large:.1%})')
```

Expected: `total_params_nli_large` en torno a 560M (aprox 2x los ~278M del backbone actual), sin errores de carga (los `UNEXPECTED` de `classifier.*` al cargar por `AutoModel` son esperables e inofensivos, igual que con los checkpoints mDeBERTa).

- [ ] **Step 6: Añadir celda de código — entrenamiento con gradient accumulation y early stopping**

```python
optimizer_nli_large = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model_nli_large.parameters()), lr=2e-5, weight_decay=0.1)

epochs_nli_large = 5
patience_nli_large = 3
total_optim_steps_large = (len(train_loader_nli_large) // grad_accum_steps_large) * epochs_nli_large
scheduler_nli_large = get_linear_schedule_with_warmup(
    optimizer_nli_large,
    num_warmup_steps=int(0.1 * total_optim_steps_large),
    num_training_steps=total_optim_steps_large)

best_val_loss_nli_large = float('inf')
nli_large_best_val_accuracy = None
epochs_no_improve_nli_large = 0

for epoch in range(epochs_nli_large):
    model_nli_large.train()
    train_loss, train_correct, train_total = 0.0, 0, 0
    optimizer_nli_large.zero_grad()
    for step, batch in enumerate(train_loader_nli_large):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        batch_labels = batch['label'].to(device)

        logits = model_nli_large(input_ids, attention_mask, token_type_ids)
        loss = criterion(logits, batch_labels) / grad_accum_steps_large
        loss.backward()

        is_last_step = (step + 1) == len(train_loader_nli_large)
        if (step + 1) % grad_accum_steps_large == 0 or is_last_step:
            torch.nn.utils.clip_grad_norm_(model_nli_large.parameters(), max_norm=1.0)
            optimizer_nli_large.step()
            scheduler_nli_large.step()
            optimizer_nli_large.zero_grad()

        train_loss += loss.item() * grad_accum_steps_large * batch_labels.size(0)
        train_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
        train_total += batch_labels.size(0)

    model_nli_large.eval()
    val_loss, val_correct, val_total = 0.0, 0, 0
    with torch.no_grad():
        for batch in val_loader_nli_large:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            batch_labels = batch['label'].to(device)

            logits = model_nli_large(input_ids, attention_mask, token_type_ids)
            loss = criterion(logits, batch_labels)

            val_loss += loss.item() * batch_labels.size(0)
            val_correct += (logits.argmax(dim=-1) == batch_labels).sum().item()
            val_total += batch_labels.size(0)

    nli_large_val_accuracy = val_correct / val_total
    avg_val_loss_nli_large = val_loss / val_total
    print(f'Epoch {epoch+1}/{epochs_nli_large} - '
          f'loss: {train_loss/train_total:.4f} - accuracy: {train_correct/train_total:.4f} - '
          f'val_loss: {avg_val_loss_nli_large:.4f} - val_accuracy: {nli_large_val_accuracy:.4f}')

    torch.save(model_nli_large.state_dict(), os.path.join(checkpoint_dir, 'nli_large_last.pt'))

    if avg_val_loss_nli_large < best_val_loss_nli_large:
        best_val_loss_nli_large = avg_val_loss_nli_large
        nli_large_best_val_accuracy = nli_large_val_accuracy
        epochs_no_improve_nli_large = 0
        torch.save(model_nli_large.state_dict(), os.path.join(checkpoint_dir, 'nli_large_best.pt'))
        print(f'  -> nuevo mejor val_loss ({best_val_loss_nli_large:.4f}), guardado en nli_large_best.pt')
    else:
        epochs_no_improve_nli_large += 1
        print(f'  -> val_loss sin mejora desde hace {epochs_no_improve_nli_large} epoch(s) (mejor: {best_val_loss_nli_large:.4f})')
        if epochs_no_improve_nli_large >= patience_nli_large:
            print(f'Early stopping: sin mejora en val_loss durante {patience_nli_large} epochs consecutivos.')
            break
```

**Vigilar durante la ejecución:** el tiempo por época. Si un solo forward+backward se acerca al nuevo umbral de `TdrDelay`, interrumpir y avisar al usuario en vez de dejar que Windows mate el proceso sin traceback (ver aviso de TDR en `CLAUDE.md`). Si aparece un `CUDA out of memory`, es capturable (a diferencia de un TDR) — reducir `batch_size_large` a 8 y subir `grad_accum_steps_large` a 4 (mismo batch efectivo) antes de reintentar.

Expected: imprime una línea `Epoch N/5 - ...` por época y al menos un `-> nuevo mejor val_loss (...)`.

- [ ] **Step 7: Añadir celda de código — recargar mejor checkpoint, evaluar y comparar contra las dos secciones anteriores**

```python
model_nli_large.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'nli_large_best.pt')))
model_nli_large.eval()

val_preds_nli_large, val_true_nli_large = [], []
with torch.no_grad():
    for batch in val_loader_nli_large:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)
        batch_labels = batch['label'].to(device)

        logits = model_nli_large(input_ids, attention_mask, token_type_ids)
        val_preds_nli_large.extend(logits.argmax(dim=-1).cpu().tolist())
        val_true_nli_large.extend(batch_labels.cpu().tolist())

lang_df_large = pd.DataFrame({'language': val_lang_nli, 'true': val_true_nli_large, 'pred': val_preds_nli_large})
lang_df_large['correct'] = lang_df_large['true'] == lang_df_large['pred']
lang_acc_large = lang_df_large.groupby('language')['correct'].agg(accuracy='mean', n='size')

comparison_large = pd.DataFrame({
    'mnli-xnli (model_nli_aug)': lang_acc_aug['accuracy'],
    '2mil7-pretrain (model_nli_pretrain)': lang_acc_pretrain['accuracy'],
    'xlm-roberta-large (model_nli_large)': lang_acc_large['accuracy'],
})
comparison_large.loc['GLOBAL'] = [nli_aug_best_val_accuracy, nli_pretrain_best_val_accuracy, nli_large_best_val_accuracy]
comparison_large
```

Expected: tabla con 3 columnas (una por modelo) + fila `GLOBAL`. `nli_large_best_val_accuracy` en un rango plausible (0.80–0.92); si `model_nli_large` no supera a `model_nli_aug`/`model_nli_pretrain`, documentarlo en una celda markdown siguiente como resultado válido (consistente con la hipótesis de que el cuello de botella no es el tamaño del modelo — ver spec), no como fallo de la tarea.

- [ ] **Step 8: Guardar el notebook y commitear**

```bash
git add watson-notebook.ipynb
git commit -m "$(cat <<'EOF'
Add xlm-roberta-large-xnli backbone experiment with VRAM-conscious training

Tests whether a genuinely larger backbone (560M vs 278M params, 24 vs 12
layers) beats model_nli_aug/model_nli_pretrain. Uses gradient accumulation
and a reduced batch size to fit 8GB VRAM, and fixes bert_encode to handle
tokenizers without token_type_ids (RoBERTa/XLM-R family).
EOF
)"
```

---

## Task 4: Elegir el mejor modelo de los tres y regenerar `submission.csv`

**Files:**
- Modify: `watson-notebook.ipynb` — editar la celda de código añadida en la Tarea 1 (Step 3), ampliando el dict `candidates`.

**Interfaces:**
- Consumes: `nli_pretrain_best_val_accuracy`, `nli_large_best_val_accuracy`, `model_name_nli_pretrain`, `model_name_nli_large`, `tokenizer_nli_large`, y todo lo ya consumido por la celda de la Tarea 1.
- Produces: `submission.csv` final, con el mejor de los tres modelos.

- [ ] **Step 1: Editar la celda de código de la Tarea 1** (la que define `candidates`) para añadir las dos entradas nuevas:

```python
candidates = {
    'model_nli_aug': (os.path.join(checkpoint_dir, 'nli_aug_best.pt'), model_name_nli,
                       tokenizer_nli, nli_aug_best_val_accuracy),
    'model_nli_pretrain': (os.path.join(checkpoint_dir, 'nli_pretrain_best.pt'), model_name_nli_pretrain,
                            tokenizer_nli, nli_pretrain_best_val_accuracy),
    'model_nli_large': (os.path.join(checkpoint_dir, 'nli_large_best.pt'), model_name_nli_large,
                         tokenizer_nli_large, nli_large_best_val_accuracy),
}

best_name = max(candidates, key=lambda name: candidates[name][3])
best_checkpoint, best_model_name, best_tokenizer, best_accuracy = candidates[best_name]
print(f'Mejor modelo: {best_name} (val_accuracy={best_accuracy:.4f})')

import gc
for var in ['model_nli_aug', 'model_nli_pretrain', 'model_nli_large']:
    if var in globals():
        del globals()[var]
gc.collect()
torch.cuda.empty_cache()

best_model = NLIModel(best_model_name, dropout=0.4, pooling='mean').float().to(device)
best_model.load_state_dict(torch.load(best_checkpoint))
best_model.eval()

test = pd.read_csv('test.csv')
test_input_final = bert_encode(test.premise.values, test.hypothesis.values, best_tokenizer)

predictions_final = []
with torch.no_grad():
    for i in range(0, len(test), batch_size):
        input_ids = test_input_final['input_ids'][i:i+batch_size].to(device)
        attention_mask = test_input_final['attention_mask'][i:i+batch_size].to(device)
        token_type_ids = test_input_final['token_type_ids'][i:i+batch_size].to(device)
        logits = best_model(input_ids, attention_mask, token_type_ids)
        predictions_final.extend(logits.argmax(dim=-1).cpu().tolist())

submission = test.id.copy().to_frame()
submission['prediction'] = predictions_final
submission.to_csv('submission.csv', index=False)
print(f'submission.csv regenerado con {best_name} (val_accuracy={best_accuracy:.4f}), {len(submission)} filas')
```

Nota: esta versión libera explícitamente los tres modelos candidatos de VRAM antes de instanciar el ganador — necesario porque para cuando se llega aquí es posible que ya no quepan los tres en 8GB simultáneamente, y porque `model_nli_pretrain` ya se había liberado en la Tarea 3, Step 4.

- [ ] **Step 2: Ejecutar la celda editada y verificar la salida**

Expected: imprime `Mejor modelo: <nombre>` con el `val_accuracy` más alto de los tres, sin `KeyError`/`RuntimeError` de dtype (los tres backbones se cargan con `.float()`, consistente con el aviso de fp16/fp32 de `CLAUDE.md`). Verificar `pd.read_csv('submission.csv').shape == (5195, 2)`.

- [ ] **Step 3: Añadir una celda markdown corta documentando el resultado final**, con los tres `val_accuracy` (`model_nli_aug`, `model_nli_pretrain`, `model_nli_large`) y cuál ganó — igual que el resto de secciones del notebook documentan sus resultados en markdown tras cada experimento.

- [ ] **Step 4: Guardar el notebook y commitear**

```bash
git add watson-notebook.ipynb
git commit -m "$(cat <<'EOF'
Pick the best of the three backbone candidates for the final submission

Extends the candidates dict from the P0 fix to include model_nli_pretrain
and model_nli_large, so submission.csv always reflects whichever model
scored highest on val_accuracy across all three experiments.
EOF
)"
```

---

## Self-Review

**Spec coverage:** P0 → Task 1 (initial) + Task 4 (final decision). P1 (vía A) → Task 2. P2 (vía B) → Task 3, incluyendo el fix de `bert_encode` y las mitigaciones de VRAM por defecto (batch reducido + gradient accumulation) exigidas en el spec. TdrDelay gate → precondition explícita al inicio de la Task 3. Fuera de alcance (C: cabeza de clasificación, D: ensemble) → correctamente no incluidos en ninguna tarea.

**Placeholder scan:** sin TBD/TODO; todo el código está completo y es el código real a insertar, no descripciones de código.

**Type/naming consistency:** verificado que `model_name_nli_pretrain`/`tokenizer_nli_large`/`nli_pretrain_best_val_accuracy`/`nli_large_best_val_accuracy` se usan con el mismo nombre en la tarea que los produce y en las tareas que los consumen (Task 4). `K` (Task 2) vs `K_large` (Task 3) son intencionalmente variables separadas — incluso siendo values iguales (`3`), Task 3 la nombra distinto para no pisar la variable global `K` que otras celdas del notebook (gradual unfreezing, etc.) podrían seguir usando más abajo si el notebook se ejecuta completo.
