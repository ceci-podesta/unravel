#!/bin/bash
cd ~/projects/unravel
uv run python -m unravel.evaluate_zero_shot \
    --head cnn \
    --outputs experiments/03_head_cnn/
