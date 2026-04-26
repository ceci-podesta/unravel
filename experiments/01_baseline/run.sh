#!/bin/bash
# Reproducir el experimento 01_baseline (zero-shot del modelo IAM
# sobre dataset español de test). Configuración por default: head=rnn,
# preprocessing con padding a la esquina superior izquierda (sin centrar).
cd ~/projects/unravel
uv run python -m unravel.evaluate_zero_shot \
    --outputs experiments/01_baseline/
