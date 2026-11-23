#!/usr/bin/env bash
set -euo pipefail

runner="$(dirname "$0")/run_ftn.py"
python "$runner" --tasks KS
python "$runner" --tasks SID
python "$runner" --tasks ER
python "$runner" --tasks KS SID
python "$runner" --tasks KS ER
python "$runner" --tasks SID ER
python "$runner" --tasks KS SID ER
