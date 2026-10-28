#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
python -m virtualcontrast.evaluate --output outputs/results
python -m virtualcontrast.routing --output outputs/results
