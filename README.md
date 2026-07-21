# Contradictory, My Dear Watson

Notebook de NLI (Natural Language Inference) en PyTorch + Hugging Face Transformers para la competición Kaggle [Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson), que consiste en clasificar pares de frases (premisa/hipótesis) en 100+ idiomas como *entailment*, *neutral* o *contradiction*.

Todo el trabajo vive en [`watson-notebook.ipynb`](watson-notebook.ipynb).

## Resultados

| Modelo | val_accuracy | Notas |
|---|---|---|
| `bert-base-multilingual-cased` (baseline) | ~65.9% | Masked-LM genérico, no preentrenado para NLI. Fine-tuning completo, 2 épocas. |
| `mDeBERTa-v3-base-mnli-xnli` + aumento datos por traducción (`model_nli_aug`) | 88.4% | |
| `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (checkpoint alternativo) | 87.2% | Experimento: más pretraining NLI no mejora sobre `model_nli_aug`. |
| `xlm-roberta-large-xnli` (backbone 2x más grande, `model_nli_large`) | **93.0%** | Mejor modelo — el que genera `submission.csv`. |

## Estructura del notebook

1. **Baseline**: fine-tuning completo de `bert-base-multilingual-cased`, 2 épocas fijas, sin congelar ninguna capa.
2. **Transfer learning con mDeBERTa**: parte de [`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli), ya fine-tuneado en MNLI+XNLI. Congela embeddings + las 9 capas inferiores del encoder, entrena solo las 3 superiores + un classifier head nuevo (7.6% de los parámetros), con warmup+decay de LR, gradient clipping y early stopping. El ajuste con más impacto en el resultado fue subir `max_len` de 50 a 128 — con 50, el tokenizer de mDeBERTa truncaba el 45.8% de los ejemplos.
3. **Análisis por idioma y augmentación por traducción**: identifica los idiomas peor servidos (ruso, tailandés, turco) y prueba aumentar datos vía traducción (`facebook/nllb-200-distilled-600M`), ensemble, domain-adaptive pretraining en tailandés, verificación de data leakage, checkpoint averaging (SWA-style) y descongelamiento gradual del encoder.

Detalles completos de cada experimento, resultados numéricos y las lecciones aprendidas (qué mejoró, qué no y por qué) en [`CLAUDE.md`](CLAUDE.md).

## Estado actual (2026-07-21)

Trabajo de mejora de accuracy completo, con `submission.csv` regenerado a partir del mejor de tres candidatos:

- **Corregido un bug real**: `submission.csv` se generaba con el modelo baseline (~65.9%), no con el mejor modelo entrenado — ninguna celda de las secciones de transfer learning volvía a generar predicciones sobre `test.csv`.
- **Experimento con checkpoint alternativo** (`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, mismo tamaño, más pretraining NLI): resultado negativo, 87.2% frente al 88.4% de `model_nli_aug`.
- **Experimento con backbone más grande** (`xlm-roberta-large-xnli`, ~560M parámetros frente a los ~278M de mDeBERTa): **93.0%**, +4.6 puntos sobre `model_nli_aug` — el mejor resultado con diferencia, y el que ahora genera `submission.csv`. Contradice la lección de los experimentos de descongelamiento gradual (más capacidad entrenable dentro del mismo backbone no movía el techo): un backbone genuinamente distinto y más grande sí lo hizo.
- El entrenamiento del backbone grande se hizo con `batch_size=8` + gradient accumulation (batch efectivo 32) para evitar el reset de driver NVIDIA (TDR) documentado en [`CLAUDE.md`](CLAUDE.md); no volvió a ocurrir tras cerrar ScreenPal (la causa real, confirmada) antes de entrenar.

## Datos

Los ficheros `train.csv`, `test.csv` y `sample_submission.csv` no están incluidos en este repo (son los datos de la competición Kaggle). Descárgalos desde la [página de datos de la competición](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data) y colócalos en la raíz del proyecto antes de ejecutar el notebook.

## Entorno

El proyecto usa un venv de Python dedicado (no conda) con las versiones fijadas en [`requirements.txt`](requirements.txt), incluyendo un build de PyTorch con soporte CUDA (`+cu126`) para entrenar en GPU. Instrucciones completas de configuración y ejecución en [`CLAUDE.md`](CLAUDE.md).
