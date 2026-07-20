# Contradictory, My Dear Watson

Notebook de NLI (Natural Language Inference) en PyTorch + Hugging Face Transformers para la competición Kaggle [Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson), que consiste en clasificar pares de frases (premisa/hipótesis) en 100+ idiomas como *entailment*, *neutral* o *contradiction*.

Todo el trabajo vive en [`watson-notebook.ipynb`](watson-notebook.ipynb).

## Resultados

| Modelo | val_accuracy | Notas |
|---|---|---|
| `bert-base-multilingual-cased` (baseline) | ~65.9% | Masked-LM genérico, no preentrenado para NLI. Fine-tuning completo, 2 épocas. |
| `mDeBERTa-v3-base-mnli-xnli` + aumento datos por traducción (`model_nli_aug`) | **88.4%** | Mejor modelo actual — el que genera `submission.csv`. |
| `mDeBERTa-v3-base-xnli-multilingual-nli-2mil7` (checkpoint alternativo) | 87.2% | Experimento: más pretraining NLI no mejora sobre `model_nli_aug`. |
| `xlm-roberta-large-xnli` (backbone 2x más grande) | ~92.8%* | Prometedor — entrenado dos veces con resultado consistente, pero aún no integrado formalmente en el notebook (ver [Estado actual](#estado-actual-2026-07-20)). |

\* Resultado real, verificado, pendiente de que la celda de comparación final se ejecute con éxito de principio a fin en el mismo notebook (ha topado dos veces con un reset de driver NVIDIA en la última celda, tras completar el entrenamiento). Los checkpoints existen en disco.

## Estructura del notebook

1. **Baseline**: fine-tuning completo de `bert-base-multilingual-cased`, 2 épocas fijas, sin congelar ninguna capa.
2. **Transfer learning con mDeBERTa**: parte de [`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli), ya fine-tuneado en MNLI+XNLI. Congela embeddings + las 9 capas inferiores del encoder, entrena solo las 3 superiores + un classifier head nuevo (7.6% de los parámetros), con warmup+decay de LR, gradient clipping y early stopping. El ajuste con más impacto en el resultado fue subir `max_len` de 50 a 128 — con 50, el tokenizer de mDeBERTa truncaba el 45.8% de los ejemplos.
3. **Análisis por idioma y augmentación por traducción**: identifica los idiomas peor servidos (ruso, tailandés, turco) y prueba aumentar datos vía traducción (`facebook/nllb-200-distilled-600M`), ensemble, domain-adaptive pretraining en tailandés, verificación de data leakage, checkpoint averaging (SWA-style) y descongelamiento gradual del encoder.

Detalles completos de cada experimento, resultados numéricos y las lecciones aprendidas (qué mejoró, qué no y por qué) en [`CLAUDE.md`](CLAUDE.md).

## Estado actual (2026-07-20)

Trabajo en curso para mejorar la puntuación real en el leaderboard de Kaggle (no solo el `val_accuracy` interno):

- **Corregido un bug real**: `submission.csv` se generaba con el modelo baseline (~65.9%), no con el mejor modelo entrenado — ninguna celda de las secciones de transfer learning volvía a generar predicciones sobre `test.csv`. Ya corregido: ahora usa `model_nli_aug` (88.4%).
- **Experimento con checkpoint alternativo** (`mDeBERTa-v3-base-xnli-multilingual-nli-2mil7`, mismo tamaño, más pretraining NLI): resultado negativo, 87.2% frente al 88.4% actual.
- **Experimento con backbone más grande** (`xlm-roberta-large-xnli`, ~560M parámetros frente a los ~278M actuales): resultado muy prometedor (~92.8%), pero la celda final de comparación aún no se ha completado con éxito en una sola sesión del notebook — dos intentos han topado con un reset de driver NVIDIA (TDR) justo después de que el entrenamiento terminara. El entrenamiento en sí es real y reproducible (verificado dos veces), y los checkpoints están guardados; falta cerrar el último paso de documentación/comparación en el notebook.

## Datos

Los ficheros `train.csv`, `test.csv` y `sample_submission.csv` no están incluidos en este repo (son los datos de la competición Kaggle). Descárgalos desde la [página de datos de la competición](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data) y colócalos en la raíz del proyecto antes de ejecutar el notebook.

## Entorno

El proyecto usa un venv de Python dedicado (no conda) con las versiones fijadas en [`requirements.txt`](requirements.txt), incluyendo un build de PyTorch con soporte CUDA (`+cu126`) para entrenar en GPU. Instrucciones completas de configuración y ejecución en [`CLAUDE.md`](CLAUDE.md).
