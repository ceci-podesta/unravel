#!/bin/bash
cd ~/projects/unravel
uv run python -m unravel.evaluate_zero_shot \
    --head both \
    --outputs experiments/04_head_both_combined/
