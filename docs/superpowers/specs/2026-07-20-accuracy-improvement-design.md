# Diseño: mejora de accuracy real (Kaggle) del notebook Watson

**Fecha:** 2026-07-20
**Estado:** aprobado, pendiente de plan de implementación

## Contexto y objetivo

El objetivo es subir la puntuación real en el leaderboard de Kaggle, no solo el
`val_accuracy` interno del notebook. Durante el brainstorming se descubrió un
hallazgo que reordena las prioridades: **`submission.csv` se genera actualmente
con el modelo baseline (~65.9% val_accuracy), no con ningún modelo de la sección
de transfer learning con mDeBERTa (~88%)**. La única celda que calcula
`predictions` sobre `test.csv` es la celda 36 (usa `model`, el baseline mBERT);
la celda 40 escribe `submission.csv` a partir de esas predicciones. Ninguna
celda posterior (mDeBERTa, augmentación, DAPT, ensemble, checkpoint averaging,
gradual unfreezing) vuelve a generar predicciones sobre `test.csv`.

Esto convierte la corrección del submission en la pieza de mayor impacto, por
delante de cualquier cambio de arquitectura.

Pregunta original que motivó el brainstorming: **¿hace falta un modelo más
potente, o hay que mejorar la cabeza de clasificación?** Evidencia ya existente
en el notebook (experimento de gradual unfreezing, `K: 0→2→4→6→8`, y el intento
descartado `K: 0→3→6→9→12` que disparó un reset de driver NVIDIA) muestra que
**más capacidad entrenable del mismo backbone no mueve el techo de accuracy**
— la ganancia vino de descongelar *progresivamente*, no de *cuánta* capacidad
se libera. Esto es evidencia en contra de invertir en la cabeza de
clasificación (ya relativamente simple: linear + pooling + dropout) como
palanca principal, y deja como pregunta abierta si un backbone *distinto*, con
mejor o más pretraining NLI, aporta algo que "más capas del mismo backbone" no
aportó.

Fuera de alcance para este ciclo (quedan documentadas para retomar si P0+P1+P2
no bastan): mejora de la cabeza de clasificación (attention pooling,
multi-sample dropout, label smoothing) y ensemble entre el modelo base y el
modelo large.

## Restricciones

- GPU: RTX 4060 Laptop, 8GB VRAM. Ya hubo un reset de driver NVIDIA (TDR) al
  descongelar `K=12` capas de un modelo *base* — ver aviso en `CLAUDE.md`.
- El usuario ha aceptado subir `HKLM\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay`
  antes de P2, para dar margen a un backbone más pesado. Esto es un cambio de
  registro de Windows que requiere reinicio — confirmar explícitamente con el
  usuario en el momento de aplicarlo, no asumirlo como ya hecho por este spec.
- Aun con TdrDelay más alto, el usuario prefiere no sobrecargar los 8GB de
  VRAM — P2 debe incluir mitigaciones de memoria por defecto, no solo como
  contingencia.
- Al ejecutar el notebook, solo correr las celdas necesarias para el objetivo
  de cada pieza (no re-ejecutar DAPT, leakage check, etc. si no cambian).

## P0 — Fix de `submission.csv` (bloqueante, primero)

Nueva celda al final del notebook (no sustituye las celdas 36/38/40 del
baseline, que se dejan intactas como documentación del flujo tutorial
original):

1. Cargar `model_nli_aug` + `checkpoints/nli_aug_best.pt` — el mejor checkpoint
   disponible hasta ahora (88.41% val_accuracy, mejor que gradual unfreezing
   88.08% y que checkpoint averaging 88.37%). Se fija a mano, sin lógica de
   selección automática, siguiendo el estilo del notebook de comparar
   resultados manualmente en markdown.
2. Tokenizar `test.csv` con el mismo `bert_encode`/`max_len=128` que el resto
   del notebook.
3. Generar `predictions` sobre `test.csv` con `model_nli_aug` (igual que la
   celda 36 original pero con este modelo).
4. Sobrescribir `submission.csv` con esas predicciones.

Si P1 o P2 producen un modelo mejor, esta celda se actualiza entonces para
apuntar al nuevo mejor checkpoint.

**Riesgo:** ninguno especial — es inferencia sobre un modelo ya entrenado.

## P1 — Vía A: swap de checkpoint pretraining (mismo tamaño)

Nueva sección con el mismo patrón que `model_nli_aug`/`model_nli_dapt` (no
sustituye nada existente):

- `model_nli_pretrain = NLIModel('MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7').float().to(device)`
  — mismo tamaño que el backbone actual (~278M params, mismo motivo de
  `.float()` que ya documenta `CLAUDE.md` para evitar el mismatch fp16/fp32),
  pero con más pretraining NLI multilingüe (~2.7M pares) que
  `mDeBERTa-v3-base-mnli-xnli`.
- Mismo esquema que `model_nli_aug` (la mejor configuración conocida): `K=3`
  capas superiores entrenables, `dropout=0.4`, `pooling='mean'`,
  `weight_decay=0.1` en el optimizer, mismo `train_loader_nli_aug` (con la
  augmentación por traducción ya incorporada) — así la única variable que
  cambia frente a `model_nli_aug` es el pretraining del backbone.
- Mismo bucle de entrenamiento con early stopping/checkpointing
  (`nli_pretrain_best.pt`/`nli_pretrain_last.pt`), recarga explícita de
  `*_best.pt`, matriz de confusión y accuracy por idioma — igual que las
  demás secciones del notebook.
- Comparación final (tabla/gráfico) de `model_nli_aug` (88.41%) vs
  `model_nli_pretrain` sobre el mismo `val_loader_nli`.

**Riesgo:** el checkpoint `2mil7` podría tener nombres de capas ligeramente
distintos en su config; al cargar solo el backbone vía `AutoModel` (sin la
classification head de ese checkpoint) debería ser un swap directo, pero se
verifica al implementar.

**Coste esperado:** bajo — mismo footprint de VRAM que `model_nli_aug`, tiempo
de entrenamiento similar (5 épocas, patience 3).

## P2 — Vía B: backbone genuinamente más grande (condicional a P1, secuencial)

Nueva sección, mismo patrón, con mitigaciones de VRAM por defecto:

- `model_nli_large = NLIModel('joeddav/xlm-roberta-large-xnli').float().to(device)`
  — backbone multilingüe grande fine-tuneado en XNLI (XLM-RoBERTa-large,
  ~550M params, ~2x el backbone actual), mismo conjunto de idiomas XNLI que
  solapa bien con el dataset de la competición.
- Mismo `K=3` capas superiores entrenables (de las 24 del encoder large — sigue
  siendo una fracción pequeña, coherente con la lección de gradual unfreezing
  de que más capacidad no es el cuello de botella), `dropout=0.4`,
  `pooling='mean'`, `weight_decay=0.1`, mismo `train_loader_nli_aug`.
- **Mitigación de VRAM por defecto** (no como contingencia opcional): batch
  size reducido respecto a `model_nli_aug`, compensado con gradient
  accumulation para mantener el batch efectivo; liberar `model_nli_pretrain`
  de memoria (`del` + `gc.collect()` + `torch.cuda.empty_cache()`, mismo
  patrón que ya usa la celda 90 del notebook) antes de cargar el modelo large.
- Contingencia documentada si sigue habiendo presión de memoria: gradient
  checkpointing (coste extra de cómputo a cambio de memoria) — no se activa
  por defecto.
- Mismo bucle con early stopping/checkpointing/matriz de confusión/accuracy
  por idioma, comparado contra `model_nli_aug` y `model_nli_pretrain`.

**Riesgo:** aunque `TdrDelay` suba, un modelo 2x más grande puede seguir
acercándose al nuevo umbral en un solo forward+backward — vigilar tiempo por
época durante el entrenamiento.

**Nota sobre expectativas:** si `model_nli_large` tampoco mejora sobre
`model_nli_pretrain`/`model_nli_aug`, sería consistente con la hipótesis de
que el cuello de botella no es el modelo en absoluto, sino ambigüedad genuina
en ciertos ejemplos (como ya sugiere el caso de tailandés tras cuatro
estrategias distintas sin efecto). Documentar el resultado en cualquier caso.

## Testing / validación

Todas las piezas reutilizan `val_idx`/`val_loader_nli` (mismo split que el
resto del notebook, para que las comparaciones sean justas) y siguen los
patrones ya establecidos: matriz de confusión (`ConfusionMatrixDisplay`,
`Blues`) y accuracy por idioma sobre el mejor checkpoint de cada modelo nuevo.
No hay ground truth en `test.csv`, así que `val_accuracy` sigue siendo el único
proxy interno disponible para la puntuación real de Kaggle.

## Fuera de alcance (documentado para retomar si hace falta)

- **Mejora de la cabeza de clasificación** (attention pooling, multi-sample
  dropout, label smoothing): descartado como prioridad por la evidencia de
  gradual unfreezing de que más capacidad no mueve el techo; el head ya es
  simple y probablemente repita ese patrón de retorno bajo.
- **Ensemble `model_nli_aug` + `model_nli_large`**: solo tiene sentido si P2
  se ejecuta. A diferencia del ensemble anterior (que falló porque los dos
  checkpoints compartían ~91% del training set con la misma arquitectura), dos
  backbones distintos deberían tener errores menos correlacionados.
