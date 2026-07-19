# Contradictory, My Dear Watson

Notebook de NLI (Natural Language Inference) en PyTorch + Hugging Face Transformers para la competición Kaggle [Contradictory, My Dear Watson](https://www.kaggle.com/competitions/contradictory-my-dear-watson), que consiste en clasificar pares de frases (premisa/hipótesis) en 100+ idiomas como *entailment*, *neutral* o *contradiction*.

Todo el trabajo vive en [`watson-notebook.ipynb`](watson-notebook.ipynb) y tiene dos secciones:

1. **Baseline**: fine-tuning completo de `bert-base-multilingual-cased` (un modelo de masked-LM genérico, no preentrenado para NLI), 2 épocas, sin congelar ninguna capa. Resultado: ~63.5% de accuracy en validación.
2. **Transfer learning con mDeBERTa**: parte de [`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli), ya fine-tuneado en MNLI+XNLI (mismo formato de tarea, 100 idiomas). Congela los embeddings y las 9 capas inferiores del encoder, y entrena solo las 3 capas superiores más un classifier head nuevo (7.6% de los 278M parámetros son entrenables), con warmup+decay de learning rate y gradient clipping. Reutiliza el mismo split train/val que el baseline para que la comparación sea justa. Resultado: ~80.7% de accuracy en validación, +17 puntos sobre el baseline.

## Datos

Los ficheros `train.csv`, `test.csv` y `sample_submission.csv` no están incluidos en este repo (son los datos de la competición Kaggle). Descárgalos desde la [página de datos de la competición](https://www.kaggle.com/competitions/contradictory-my-dear-watson/data) y colócalos en la raíz del proyecto antes de ejecutar el notebook.

## Entorno

El proyecto usa un venv de Python dedicado (no conda) con las versiones fijadas en [`requirements.txt`](requirements.txt), incluyendo un build de PyTorch con soporte CUDA (`+cu126`) para entrenar en GPU. Instrucciones completas de configuración y ejecución en [`CLAUDE.md`](CLAUDE.md).
