#!/bin/bash
cd ~/projects/unravel
uv run python -m unravel.evaluate_zero_shot \
    --center \
    --head cnn \
    --outputs experiments/05_centered_cnn/
