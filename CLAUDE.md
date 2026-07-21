# Watson (Contradictory, My Dear Watson)

Notebook de NLI (PyTorch + Hugging Face Transformers + BERT multilingüe) para la competición Kaggle "Contradictory, My Dear Watson". Vive en `watson-notebook.ipynb`, con `train.csv`, `test.csv`, `sample_submission.csv` en el mismo directorio.

El notebook tiene dos secciones:
1. **Baseline**: fine-tuning completo de `bert-base-multilingual-cased` (masked-LM genérico, no preentrenado para NLI), 2 épocas fijas, ~65.9% val_accuracy. Sin freezing de ninguna capa.
2. **Transfer learning con mDeBERTa** (al final del notebook): parte de [`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli), ya fine-tuneado en MNLI+XNLI (mismo formato de tarea, 100 idiomas). Congela embeddings + las 9 capas inferiores del encoder, entrena solo las 3 superiores + un classifier head nuevo (7.6% de los 278M parámetros son entrenables), con warmup+decay de LR, gradient clipping y hasta 15 épocas con early stopping (ver abajo). Resultado: ~88.0% val_accuracy, +22 puntos sobre el baseline (en la última ejecución, early stopping cortó en la época 7 tras 5 épocas sin mejora, quedándose con los pesos de la época 2). Reutiliza el mismo split train/val (`train_idx`/`val_idx`) que el baseline para que la comparación sea justa; no modifica ni sustituye el baseline.

### `max_len`: el cuello de botella real

`max_len` empezó en 50, heredado del baseline original. Con el tokenizer de mDeBERTa eso truncaba el **45.8%** de los pares premise+hypothesis (longitud mediana real: 48 tokens), cortando a menudo el hypothesis entero — la mitad de la información que el modelo necesita para clasificar. Subir `max_len` a 128 (cubre el 99.4% de los ejemplos sin truncar) fue, con diferencia, el cambio que más impactó el resultado: mDeBERTa pasó de ~81.2% a ~88.0% val_accuracy, y el baseline también mejoró (de ~64% a ~65.9%). Los ajustes de regularización (dropout, weight_decay, mean pooling) que se probaron antes de este fix dieron mejoras de solo décimas de punto en comparación — la lección es que con casi la mitad del dataset truncado, ningún ajuste de hiperparámetros iba a mover mucho la aguja. `max_len` es una constante compartida por `bert_encode`, así que el fix aplica a ambas secciones para mantener la comparación justa.

### Reproducibilidad y regularización de mDeBERTa

El baseline mostró ~1.3 puntos de varianza en val_accuracy entre ejecuciones idénticas (ruido del shuffling del `DataLoader` y la inicialización aleatoria del classifier nuevo), así que hay una celda de semilla fija (`SEED = 42`, justo después de los imports) que fija `random`, `numpy` y `torch`/CUDA antes de crear cualquier modelo o dataloader — no garantiza determinismo bit-exacto en GPU (cuDNN), pero elimina la mayor parte de esa varianza para que las comparaciones entre configuraciones sean fiables.

`NLIModel` (compartida por baseline y mDeBERTa) acepta `dropout` (default `0.0`) y `pooling` (`'cls'` por defecto, o `'mean'` para promediar `last_hidden_state` sobre los tokens no-padding usando `attention_mask`) — ambos parámetros opcionales para que el baseline quede intacto. La instancia de mDeBERTa usa `dropout=0.4`, `weight_decay=0.1` en el optimizer y `pooling='mean'`; se llegó a esta combinación iterando sobre el overfitting observado (train_accuracy subiendo hasta ~90% mientras val_loss empeoraba tras la época 2-3 con solo CLS pooling y sin regularizar). El overfitting post-mejor-época sigue presente incluso con `max_len=128` (train_accuracy ~93% en la época 7 vs ~87-88% val), pero el early stopping ya lo neutraliza quedándose con los pesos de la mejor época.

### Early stopping y checkpointing

Ambos bucles de entrenamiento (baseline y mDeBERTa) guardan los pesos del modelo en `checkpoints/` (no versionado, ver `.gitignore`) tras cada época:
- `{baseline,nli}_last.pt` se sobrescribe siempre con los pesos de la última época.
- `{baseline,nli}_best.pt` solo se sobrescribe cuando la `val_loss` mejora respecto a la mejor vista hasta el momento.
- Early stopping con `patience=5`: si la `val_loss` no mejora durante 5 épocas consecutivas, el bucle corta y se queda con `*_best.pt`. El baseline sigue fijado en 2 épocas (por debajo del patience, así que en la práctica no llega a activarse); mDeBERTa tiene hasta 15 épocas disponibles para que el early stopping tenga margen real de actuar.
- Tras entrenar, cada sección recarga explícitamente `*_best.pt` (no los pesos en memoria de la última época) antes de calcular la matriz de confusión y las métricas finales — así el resultado reportado corresponde siempre al mejor checkpoint, no al último.

### Matriz de confusión

Al ser NLI un problema de clasificación de 3 clases (entailment / neutral / contradiction), cada sección incluye una matriz de confusión (`sklearn.metrics.ConfusionMatrixDisplay`, cmap `Blues`) sobre el split de validación, calculada con los pesos del mejor checkpoint (`*_best.pt`), justo después de su bucle de entrenamiento.

### Augmentación por traducción, domain-adaptive pretraining y verificación de leakage

Tras el baseline y mDeBERTa, el notebook añade un análisis de accuracy por idioma sobre el split de validación (mismo `val_idx` que el resto), que identificó ruso, tailandés y turco como los tres idiomas peor servidos (80.5%/81.2%/81.5%, todos por debajo del 87.7% global). Se probaron tres estrategias sobre esos tres idiomas, todas midiendo el efecto sobre el mismo `val_loader_nli`:

1. **Augmentación por traducción** (`facebook/nllb-200-distilled-600M`, ~300 ejemplos por idioma traducidos desde inglés, label preservada): ruso +4.9pts (80.5%→85.4%), turco +1.9pts, **tailandés sin cambio** (81.2%→81.2%).
2. **Ensemble** (promedio de softmax entre el modelo con y sin augmentación): no supera al mejor modelo individual — los dos checkpoints comparten ~91% del training set, así que sus errores están demasiado correlacionados para que promediar aporte algo.
3. **Domain-adaptive pretraining (DAPT)** en tailandés: MLM continuado (`AutoModelForMaskedLM`, mismas `K=3` capas superiores entrenables) sobre un corpus de 913 frases tailandesas reales sin traducir (texto de `train_idx` + `test.csv`, nunca de `val_idx`), seguido de fine-tuning supervisado normal. Tampoco movió tailandés (81.16% exacto, idéntico a las otras dos estrategias) — dos vías completamente distintas convergiendo al mismo resultado nulo sugiere que el cuello de botella en tailandés no es de representación/dominio, sino probablemente ambigüedad genuina en los propios ejemplos; pendiente de inspección manual antes de seguir invirtiendo en más datos o más pretraining para ese idioma.

Se verificó también **data leakage** entre `train_idx`/`val_idx`: cero duplicados exactos `premise+hypothesis`, pero un 47.3% de las premises de validación también aparecen en train con otra hypothesis/label (el dataset genera hasta 3 filas por premise y el split no agrupa por ese campo). Comparando accuracy entre el subconjunto con esa fuga y el subconjunto limpio salió prácticamente idéntico (87.5% vs 87.9%), así que **no infla las cifras reportadas** — no fue necesario rehacer el split.

Dos mejoras más, ambas orientadas al procedimiento de entrenamiento en vez de a los datos:

- **Checkpoint averaging (SWA-style)** sobre `model_nli_aug`: promediar los pesos de las épocas 2, 3 y 4 (la mejor fue la 3) dio 88.37% frente al 88.41% del mejor checkpoint solo — un empate técnico, ligeramente negativo. Con un LR schedule que decae (no constante/cíclico como requiere SWA para funcionar bien), las épocas cercanas a la mejor ya están demasiado convergidas entre sí como para que promediarlas aporte diversidad real.
- **Gradual unfreezing**: en vez de fijar `K=3` capas entrenables desde la época 1, se activa progresivamente (mismo presupuesto de 5 épocas que `model_nli_aug`, mismo optimizer construido una sola vez sobre todos los parámetros para no perder momentum al cambiar `requires_grad`). Rampa final `K: 0→2→4→6→8` (un primer intento con `K: 0→3→6→9→12`, hasta el encoder completo, provocó un **reset del driver de NVIDIA por TDR** — ver aviso más abajo — así que se recortó). Resultado: 87.71%→88.08% (+0.37pts) sin tocar los datos, idéntico al resultado obtenido con una rampa más corta (tope `K=3`) — confirma, igual que ya hizo el `K=6` estático, que más capacidad entrenable no mueve el techo de esta tarea; la ganancia viene de introducir el descongelamiento *progresivo* en sí, no de cuánto se acaba liberando. El mejor checkpoint fue el de la época 3 (`K=4`), no el final: las épocas con `K` más alto ya sobreajustaban. Ganancias en ruso (+2.4pts) y turco (+1.9pts) parecidas en magnitud a las de la augmentación por traducción, pero por una vía distinta (procedimiento de entrenamiento, no datos). Tailandés bajó a 79.7% en esta rampa más larga (antes se quedaba plano en 81.2% con las otras tres estrategias) — ya son cuatro intentos distintos que no lo mejoran, lo que apunta a ambigüedad genuina en esos ejemplos más que a un problema de datos/representación/procedimiento.

### Aviso: reset de driver de NVIDIA (TDR) con cargas de entrenamiento pesadas

Entrenar con `K` alto (p.ej. descongelar el encoder completo, `K=12`) puede disparar un **reset del driver de NVIDIA** en este equipo — Windows fuerza un reset de la GPU (TDR, Timeout Detection and Recovery) si un kernel CUDA individual tarda más del umbral por defecto de 2 segundos sin responder, y el backward completo por las 12 capas de un modelo de disentangled attention como DeBERTa-v3 puede superarlo. Se manifiesta como un proceso Python matado sin excepción capturable ni traceback (a diferencia de un `OutOfMemoryError`, que sí es capturable) — para diagnosticarlo, comprobar en el Visor de sucesos de Windows (`Get-WinEvent -FilterHashtable @{LogName='System'; ProviderName='nvlddmkm'}`) un evento de error ID 153 con el mismo timestamp que el corte. El fix estándar es subir `HKLM\\SYSTEM\\CurrentControlSet\\Control\\GraphicsDrivers\\TdrDelay` (DWORD, segundos; por defecto 2) a un valor mayor (p.ej. 60) y reiniciar — no aplicado en este repo por decisión explícita del usuario; en su lugar, evitar cargas de entrenamiento tan pesadas (`K` más bajo, batch más pequeño) es la mitigación usada aquí.

## Ejecutar el notebook con GPU

Este proyecto tiene un **venv de Python dedicado** en `.venv/` (creado con `python -m venv` + `pip install -r requirements.txt`), no un entorno conda. Se intentó usar conda (`environment.yml`) pero los canales de Anaconda (`conda.anaconda.org`, `repo.anaconda.com`) fallan con un error de verificación de certificado SSL en esta máquina (probablemente proxy/antivirus interceptando TLS) — `pip`/PyPI sí funcionan sin problema, de ahí el cambio a venv. Si en el futuro el problema de certificados de conda se resuelve, se podría volver a un `environment.yml`, pero de momento el venv es la vía soportada.

El notebook tiene su `kernelspec` apuntando a `watson-nli` (el kernel registrado desde este venv), así que tanto abrirlo en Jupyter como ejecutarlo con `nbconvert` usará el venv correcto por defecto.

Antes de ejecutar el notebook:

1. **Verificar que la GPU esté disponible** en el sistema:
   ```
   nvidia-smi
   ```
   Debe mostrar la GPU (RTX 4060) con driver y CUDA runtime listados. Si no aparece, avisar al usuario antes de continuar — no ejecutar el notebook asumiendo CPU en su lugar sin decírselo.

2. **Asegurar que existe el venv** en `.venv/`. Si no existe todavía:
   ```
   python -m venv .venv
   .venv\Scripts\python.exe -m pip install --upgrade pip
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   .venv\Scripts\python.exe -m ipykernel install --user --name watson-nli --display-name "watson-nli (GPU)"
   ```
   Si ya existe pero `requirements.txt` ha cambiado desde la última vez, reinstalar con:
   ```
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   ```

3. **Ejecutar el notebook usando el kernel/venv `watson-nli`**, no el Python global ni el conda `base`:
   ```
   .venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace watson-notebook.ipynb
   ```
   Esto puede tardar varios minutos (descarga del modelo BERT + 2 épocas de entrenamiento). Lanzarlo en background y avisar cuando termine.

4. **Confirmar tras la ejecución** que la celda del device (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`) imprimió `Using device: cuda` y no `cpu`. Si salió `cpu`, algo falló en el entorno (torch instalado sin build CUDA) — no dar la ejecución por buena en ese caso, investigar antes de reportar éxito.

## Descargas de Hugging Face (proxy corporativo)

Esta máquina está detrás de un proxy/antivirus que hace TLS-interception. `urllib` confía en el almacén de certificados de Windows y funciona sin problema, pero `httpx`/`huggingface_hub` (usado por `transformers` para descargar modelos) usa el bundle de `certifi` por defecto y falla con `CERTIFICATE_VERIFY_FAILED` en descargas nuevas (no cacheadas). El notebook soluciona esto con `truststore.inject_into_ssl()` al principio (cell de imports), que hace que Python confíe en el almacén de certificados del sistema — la solución correcta, no equivale a desactivar la verificación SSL. Si se añade código nuevo que descargue de HF Hub fuera del notebook (scripts sueltos, etc.), añadir las mismas dos líneas al principio:
```python
import truststore
truststore.inject_into_ssl()
```

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

## Cuidado con la precisión (fp16 vs fp32) al cargar modelos nuevos

Algunos checkpoints de Hugging Face (p.ej. `mDeBERTa-v3-base-mnli-xnli`) están guardados en fp16, y `AutoModel.from_pretrained` carga los pesos en esa precisión por defecto. Si se combina con una capa nueva en fp32 (como el `classifier` de `NLIModel`), falla con `RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float`. Al instanciar un backbone nuevo, forzar fp32 explícitamente: `NLIModel(model_name).float().to(device)`.

## Mantenimiento del entorno

- Las versiones de las librerías están fijadas en `requirements.txt` (torch, transformers, sentencepiece, protobuf, truststore, pandas, numpy, scikit-learn, matplotlib, jupyter/nbconvert/notebook/jupyterlab). El build de torch es `+cu126` (compilado con soporte CUDA) — instalar torch sin especificar `--extra-index-url https://download.pytorch.org/whl/cu126` da por defecto la versión CPU-only y rompe la GPU en silencio. `sentencepiece`/`protobuf` son necesarios para tokenizers tipo DeBERTa-v3/XLM-R.
- El driver NVIDIA y el runtime CUDA son a nivel de sistema operativo, no del venv — no se fijan en estos ficheros.
- `.venv/` no debería subirse a control de versiones si en el futuro se añade git (añadir a `.gitignore`).
