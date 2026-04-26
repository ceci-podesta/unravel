#!/bin/bash
cd ~/projects/unravel
uv run python -m unravel.evaluate_zero_shot \
    --center \
    --outputs experiments/02_centered/
