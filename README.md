# unravel

Reconocimiento de texto manuscrito en letra cursiva y en idioma español.
Adaptación de un modelo HTR preentrenado en inglés (IAM) al dominio del
español manuscrito mediante fine-tuning con LoRA.

## Setup

Requisitos:
- Python 3.12
- `uv` (gestor de entornos): https://github.com/astral-sh/uv
- GPU NVIDIA con CUDA (recomendado; CPU funciona pero lento)

```bash
git clone https://github.com/ceci-podesta/unravel.git
cd unravel
uv sync --all-extras
uv run pytest -v   # verificar que todo está OK (60 tests)
```

## Recursos externos a bajar

### Modelo preentrenado IAM

Repo de referencia: <https://github.com/georgeretsi/HTR-best-practices>

```bash
cd ~/projects   # o donde clones repos
git clone https://github.com/georgeretsi/HTR-best-practices.git
```

Necesitamos dos archivos del repo:
- `HTR-best-practices/saved_models/htrnet.pt` — pesos preentrenados (~30 MB).
- `HTR-best-practices/saved_models/classes.npy` — vocabulario IAM (79 chars).

Los scripts asumen que el repo está en `~/projects/HTR-best-practices/`
por default. Se puede cambiar con los flags `--weights` y `--classes`.

### Dataset español

Source: <https://www.kaggle.com/datasets/verack/spanish-handwritten-characterswords>

Requiere cuenta de Kaggle. Estructura esperada después de bajar y descomprimir:

```
~/datasets/spanish-htr/
├── datos_entrenamiento/PERFECT_CUT_a_z_1_9/
│   ├── *.jpg                 (~6.100 imágenes)
│   └── 0annotation.json      (filename → palabra)
├── datos_entrenamiento_augmented/PERFECT_CUT_a_z_1_9_aug_SYNTHETIC/
│   ├── *.jpg                 (~188.000 imágenes sintéticas, NO USADO actualmente)
│   └── 0annotation.json      (con mojibake conocido en 315 entries)
└── datos_testing/
    ├── 1/, 2/, ..., 11/      (organizado por longitud de palabra)
    └── gt_*.txt              (TSV: filename ↔ palabra)
```

Si están en otra ruta, los scripts aceptan `--dataset PATH`.

### Para reality check (opcional)

IAM word-level: <https://www.kaggle.com/datasets/ngkinwang/iam-dataset/data>

Estructura esperada después de bajar:

```
~/datasets/iam_word_level/
├── linux_gt.txt
├── train_gt.txt
├── val_gt.txt
└── words/<carpeta>/<subcarpeta>/<filename>.png
```

## Cómo ejecutar

### Zero-shot evaluation (sin fine-tuning)

```bash
uv run python -m unravel.evaluate_zero_shot \
    --outputs experiments/01_baseline/
```

Flags relevantes:

| Flag | Default | Descripción |
|---|---|---|
| `--center` | (off) | Centra la palabra en el canvas en lugar de pegarla a la esquina sup-izq. |
| `--head` | `rnn` | `rnn` / `cnn` / `both`. Cuál head del modelo usar en inferencia. `cnn` fue el mejor en zero-shot. |
| `--batch-size` | `16` | Tamaño de batch para inferencia. |
| `--dataset` | `~/datasets/spanish-htr/datos_testing` | Path al test set. |
| `--weights` | `~/projects/HTR-best-practices/saved_models/htrnet.pt` | Pesos del modelo. |
| `--classes` | `~/projects/HTR-best-practices/saved_models/classes.npy` | Vocabulario IAM. |
| `--outputs` | `~/projects/unravel/outputs/zero_shot_test` | Directorio de salida (predictions.csv + summary.json). |

Cada experimento publicado vive en `experiments/0X_*/` con su `run.sh`
para reproducirlo y un `notes.md` con la hipótesis y lectura.

### LoRA fine-tuning

```bash
uv run python -m unravel.train_lora \
    --outputs experiments/06_lora_real_only/
```

Flags clave para experimentación:

| Flag | Default | Descripción |
|---|---|---|
| `--r` | `8` | Rango LoRA. Más alto = más capacidad de adaptación, más riesgo de overfitting. Probar 16 si hay plateau. |
| `--alpha` | `16` | Factor de escala α. Convención: `α = 2·r`. |
| `--lora-dropout` | `0.1` | Regularización interna del LoRA. Subir hasta 0.2 si hay overfitting visible. |
| `--lr` | `5e-4` | Tasa de aprendizaje. Para LoRA conviene más alta que para full fine-tune. |
| `--head` | `both` | `rnn` / `cnn` / `both`. En training, `both` entrena ambas cabezas en paralelo. |
| `--epochs` | `10` | Cantidad de epochs. Si val_loss sigue bajando al final, extender. |
| `--batch-size` | `16` | Bajar a 8 si hay `CUDA out of memory`. |
| `--num-workers` | `2` | Workers del DataLoader. |
| `--max-train-batches` | `None` | Limita batches por epoch (útil para dry-runs). |
| `--max-val-batches` | `None` | Idem para validación. |
| `--seed` | `42` | Seed para la partición train/val (90/10) — fijo para reproducibilidad. |

#### Hiperparámetros que más mueven la aguja

1. **`r`** (rango): de 8 a 16 suele dar mejora notable si hay plateau.
2. **`lr` + `epochs`**: ajustar juntos. Si hay mejora gradual al final
   del training, extender `--epochs`. Si oscila mucho, bajar `lr`.
3. **`head`**: `cnn` fue mejor que `rnn` en zero-shot. En training con
   LoRA, `both` permite que ambas se adapten en paralelo.

#### Outputs del training

```
experiments/06_lora_real_only/
├── metrics_per_step.csv      # loss por step
├── metrics_per_epoch.csv     # train/val loss + CER + WER por epoch
├── summary.json              # config, lora_stats, history, tiempo total
├── loss_curve.png            # train vs val loss
├── cer_wer_curve.png         # CER y WER en val por epoch
├── training.log              # output crudo del run
└── best_lora/                # pesos LoRA del mejor epoch (val_CER mínimo)
    ├── adapter_config.json
    └── adapter_model.safetensors
```

## Tests

```bash
uv run pytest -v               # todos
uv run pytest tests/test_metrics.py -v   # solo métricas
```

Total: 60 tests. Si algo rompe, no hacer release.

## Estructura del repo

```
unravel/
├── src/unravel/
│   ├── htr_model.py          # HTRNet (CNN + BiLSTM + CTC)
│   ├── preproc.py            # load_image + preprocess (con fix RGB)
│   ├── dataset.py            # SpanishHTRTestDataset (test set)
│   ├── train_dataset.py      # SpanishHTRTrainDataset (split 90/10)
│   ├── vocab.py              # Vocabulario unificado IAM + español
│   ├── extend_vocab.py       # Extiende capas de salida del modelo
│   ├── ctc_utils.py          # Collate function + CTC loss helper
│   ├── lora_setup.py         # Wrapper de PEFT/LoRA
│   ├── evaluate_zero_shot.py # Eval zero-shot
│   ├── train_lora.py         # Training script de fine-tuning LoRA
│   ├── explore.py            # Inspección del dataset
│   └── metrics.py            # CER (micro/macro), WER, edit_distance
├── tests/                    # pytest, 60 tests
├── experiments/0X_*/         # cada experimento con run.sh + notes.md + outputs
├── pyproject.toml            # gestionado con uv
└── uv.lock                   # lockfile (versiones exactas)
```

## Referencias

- **LoRA**: Hu, E. J. et al. (2021). *LoRA: Low-Rank Adaptation of Large Language Models.* arXiv:[2106.09685](https://arxiv.org/abs/2106.09685)
- **Modelo base HTR**: Retsinas, G. et al. (2022). *Best Practices for a Handwritten Text Recognition system.* DAS. Repo: <https://github.com/georgeretsi/HTR-best-practices>
- **Generalización HTR (motivación del TP)**: Garrido-Muñoz et al. (2025). *On the Generalization of HTR Models.* CVPR.
- **CTC tutorial visual**: Hannun, A. (2017). *Sequence Modeling with CTC.* Distill. <https://distill.pub/2017/ctc/>
- **CTC paper original**: Graves, A. et al. (2006). *Connectionist Temporal Classification.* ICML. <https://www.cs.toronto.edu/~graves/icml_2006.pdf>
- **Dataset español**: <https://www.kaggle.com/datasets/verack/spanish-handwritten-characterswords>
- **Dataset IAM word-level** (opcional, para reality check): <https://www.kaggle.com/datasets/ngkinwang/iam-dataset/data>
