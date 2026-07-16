# Watson (Contradictory, My Dear Watson)

Notebook de NLI (PyTorch + Hugging Face Transformers + BERT multilingüe) para la competición Kaggle "Contradictory, My Dear Watson". Vive en `tutorial-notebook.ipynb`, con `train.csv`, `test.csv`, `sample_submission.csv` en el mismo directorio.

El notebook tiene dos secciones:
1. **Baseline**: fine-tuning completo de `bert-base-multilingual-cased` (masked-LM genérico, no preentrenado para NLI), 2 épocas, ~63.5% val_accuracy. Sin freezing de ninguna capa.
2. **Transfer learning con mDeBERTa** (al final del notebook): parte de [`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`](https://huggingface.co/MoritzLaurer/mDeBERTa-v3-base-mnli-xnli), ya fine-tuneado en MNLI+XNLI (mismo formato de tarea, 100 idiomas). Congela embeddings + las 9 capas inferiores del encoder, entrena solo las 3 superiores + un classifier head nuevo (7.6% de los 278M parámetros son entrenables), con warmup+decay de LR y gradient clipping. Resultado: ~80.7% val_accuracy, +17 puntos sobre el baseline. Reutiliza el mismo split train/val (`train_idx`/`val_idx`) que el baseline para que la comparación sea justa; no modifica ni sustituye el baseline.

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
   .venv\Scripts\jupyter.exe nbconvert --to notebook --execute --inplace tutorial-notebook.ipynb
   ```
   Esto puede tardar varios minutos (descarga del modelo BERT + 2 épocas de entrenamiento). Lanzarlo en background y avisar cuando termine.

4. **Confirmar tras la ejecución** que la celda del device (`torch.device('cuda' if torch.cuda.is_available() else 'cpu')`) imprimió `Using device: cuda` y no `cpu`. Si salió `cpu`, algo falló en el entorno (torch instalado sin build CUDA) — no dar la ejecución por buena en ese caso, investigar antes de reportar éxito.

## Descargas de Hugging Face (proxy corporativo)

Esta máquina está detrás de un proxy/antivirus que hace TLS-interception. `urllib` confía en el almacén de certificados de Windows y funciona sin problema, pero `httpx`/`huggingface_hub` (usado por `transformers` para descargar modelos) usa el bundle de `certifi` por defecto y falla con `CERTIFICATE_VERIFY_FAILED` en descargas nuevas (no cacheadas). El notebook soluciona esto con `truststore.inject_into_ssl()` al principio (cell de imports), que hace que Python confíe en el almacén de certificados del sistema — la solución correcta, no equivale a desactivar la verificación SSL. Si se añade código nuevo que descargue de HF Hub fuera del notebook (scripts sueltos, etc.), añadir las mismas dos líneas al principio:
```python
import truststore
truststore.inject_into_ssl()
```

## Cuidado con la precisión (fp16 vs fp32) al cargar modelos nuevos

Algunos checkpoints de Hugging Face (p.ej. `mDeBERTa-v3-base-mnli-xnli`) están guardados en fp16, y `AutoModel.from_pretrained` carga los pesos en esa precisión por defecto. Si se combina con una capa nueva en fp32 (como el `classifier` de `NLIModel`), falla con `RuntimeError: mat1 and mat2 must have the same dtype, but got Half and Float`. Al instanciar un backbone nuevo, forzar fp32 explícitamente: `NLIModel(model_name).float().to(device)`.

## Mantenimiento del entorno

- Las versiones de las librerías están fijadas en `requirements.txt` (torch, transformers, sentencepiece, protobuf, truststore, pandas, numpy, scikit-learn, matplotlib, jupyter/nbconvert/notebook/jupyterlab). El build de torch es `+cu126` (compilado con soporte CUDA) — instalar torch sin especificar `--extra-index-url https://download.pytorch.org/whl/cu126` da por defecto la versión CPU-only y rompe la GPU en silencio. `sentencepiece`/`protobuf` son necesarios para tokenizers tipo DeBERTa-v3/XLM-R.
- El driver NVIDIA y el runtime CUDA son a nivel de sistema operativo, no del venv — no se fijan en estos ficheros.
- `.venv/` no debería subirse a control de versiones si en el futuro se añade git (añadir a `.gitignore`).
