#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_root"
export CUDA_VISIBLE_DEVICES=""
export MPLBACKEND=Agg

python scripts/verify_reference.py
python scripts/audit_privacy.py . --trust-local-weights --expect-checkpoints 7 --exclude-generated
python -m unittest discover -s tests -v
python scripts/plot_results.py --input results --output reproduced/figures
