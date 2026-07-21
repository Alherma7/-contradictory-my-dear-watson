# Diseño: notebook de submission para Kaggle (`kaggle-submission.ipynb`)

**Fecha:** 2026-07-21
**Estado:** aprobado, pendiente de plan de implementación

## Contexto y objetivo

`watson-notebook.ipynb` ya genera `submission.csv` localmente a partir del
mejor de tres backbones comparados (`model_nli_large`, `xlm-roberta-large-xnli`,
92.99% val_accuracy — ver [`README.md`](../../../README.md)). Sin embargo, esta
competición de Kaggle ("Contradictory, My Dear Watson") no acepta la subida
directa de un CSV: exige un **notebook ejecutado dentro del entorno de
Kaggle** ("Save & Run All") cuyo output sea `submission.csv`, seguido de
"Submit to Competition" desde ahí.

El notebook local no es apto para correr tal cual en Kaggle: asume un venv
local, un proxy corporativo con TLS-interception (`truststore`), y
checkpoints de varias horas de entrenamiento en disco. El objetivo de este
ciclo es un notebook nuevo, separado, mínimo, que se pueda subir a Kaggle y
producir `submission.csv` sin reentrenar nada.

## Decisión de alcance (tomada en brainstorming)

Dos preguntas resueltas explícitamente con el usuario:

1. **¿Replicar todo el pipeline experimental o solo el camino ganador?** →
   Solo el camino ganador, en un notebook nuevo y ligero, **reutilizando los
   pesos ya entrenados** (`checkpoints/nli_large_best.pt`) en vez de
   reentrenar en Kaggle.
2. **¿Replicar la augmentación por traducción (NLLB) que se usó para
   entrenar ese checkpoint?** → No hace falta: al reutilizar los pesos ya
   entrenados, la augmentación ya está "horneada" en el checkpoint. El
   notebook de Kaggle es **solo inferencia**.

Esto descarta cualquier entrenamiento, augmentación, comparación de
candidatos o checkpoint averaging del notebook de Kaggle — nada de eso hace
falta para cargar un modelo ya elegido como ganador y predecir sobre
`test.csv`.

## Restricción: el checkpoint no está en Kaggle

`nli_large_best.pt` pesa 2.24GB y vive solo en `checkpoints/` (no
versionado, ver `.gitignore`). Kaggle no tiene acceso al disco local, así
que debe subirse como un **Kaggle Dataset privado** y adjuntarse al
notebook como input antes de ejecutarlo.

No hay credenciales de Kaggle API (`kaggle.json`) configuradas en esta
máquina, así que esa subida es manual (vía la web de Kaggle, "New
Dataset" → arrastrar el archivo) — no automatizable desde aquí en este
ciclo. El notebook documenta en su primera celda markdown el nombre de
archivo exacto que espera y los dos pasos de configuración manual
requeridos en Kaggle antes de ejecutar:

- Adjuntar el dataset con el checkpoint (`Add Input` → el dataset subido).
- Activar Internet en `Settings` del notebook (necesario para descargar
  tokenizer + arquitectura base de `joeddav/xlm-roberta-large-xnli` desde
  Hugging Face Hub; los pesos fine-tuneados se sobrescriben después con
  `load_state_dict` desde el checkpoint adjunto).

## Contenido del notebook

Adaptado de las celdas 106-112 de `watson-notebook.ipynb`, sin nada
específico del entorno local (venv, proxy, `truststore`, gestión de
`checkpoints/` de varios modelos):

1. **Markdown de instrucciones**: pasos manuales de Kaggle (dataset +
   Internet ON) antes de ejecutar, y qué nombre de archivo espera el
   notebook para el checkpoint.
2. **Imports + `SEED = 42`** (misma semilla que el resto del proyecto, por
   consistencia — aunque en inferencia pura con el modelo en `.eval()` no
   hay aleatoriedad de entrenamiento que fijar).
3. **Carga de `test.csv`** desde `/kaggle/input/contradictory-my-dear-watson/test.csv`
   (ruta estándar de Kaggle para los datos adjuntos de la competición).
4. **`NLIModel`** (clase idéntica a la del notebook local: backbone +
   pooling `'cls'`/`'mean'` + classifier lineal) y **`bert_encode`**
   (`max_len=128`, idéntico — mismo motivo documentado en `CLAUDE.md`:
   con 50 se truncaba el 45.8% de los ejemplos).
5. **Carga del modelo**: tokenizer + backbone base
   `joeddav/xlm-roberta-large-xnli` vía `AutoTokenizer`/`AutoModel`,
   instanciado como `NLIModel(model_name, dropout=0.4, pooling='mean').float().to(device)`
   (fp32 explícito — el mismo cuidado fp16/fp32 documentado en `CLAUDE.md`),
   después `load_state_dict(torch.load(<ruta del checkpoint adjunto>))`,
   `.eval()`.
6. **Inferencia por batches** sobre `test.csv` (sin gradientes, `batch_size`
   más generoso que en entrenamiento ya que no hay backward ni gradient
   accumulation que gestionar).
7. **Construcción y escritura de `submission.csv`** (columnas `id`,
   `prediction`, mismo formato que `sample_submission.csv`) en
   `/kaggle/working/`, que es lo que Kaggle recoge automáticamente al hacer
   "Submit to Competition" tras "Save & Run All".

## Fuera de alcance

- Reentrenar cualquier backbone dentro de Kaggle.
- Augmentación por traducción, DAPT, comparación de candidatos, checkpoint
  averaging, gradual unfreezing — todo eso vive solo en el notebook local
  de experimentación.
- Automatizar la subida del checkpoint como Kaggle Dataset (requeriría
  `kaggle.json`, no configurado; queda como paso manual documentado).
- Verificar en vivo el resultado real en el leaderboard de Kaggle (requiere
  que el usuario suba el dataset y ejecute el notebook en Kaggle).
