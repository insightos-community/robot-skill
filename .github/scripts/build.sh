#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail
mkdir -p .output/payload
uv venv .output/test --python 3.13
uv pip install --python .output/test/bin/python 'pytest>=8,<9' 'pydantic>=2.8,<3' 'numpy>=2,<3' 'scipy>=1.14,<2'
PATH="$PWD/.output/test/bin:$PATH" make check test
uv build
python3 ../automation/.github/scripts/skills.py
