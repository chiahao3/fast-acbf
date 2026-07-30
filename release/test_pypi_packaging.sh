#!/bin/bash
# test_pypi_packaging.sh - Sanity check for PyPI releases
#
# Builds the wheel, installs it into a clean conda env, and verifies
# `import fast_acbf` works and reports the expected version. Mirrors
# ptyrad's release/test_pypi_packaging.sh, trimmed down since fast-acbf
# has no CLI entry point to exercise.

set -e

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

## 0. Ensure build is installed
if ! python -m pip show build &> /dev/null; then
    echo "❌ ERROR: 'build' is not installed. Run 'pip install build' first."
    exit 1
fi

## 1. Grab the expected version before touching anything else
EXPECTED_VERSION=$(python -c "import re; print(re.search(r'__version__\s*=\s*\"([^\"]+)\"', open('src/fast_acbf/__init__.py').read()).group(1))")
echo "ℹ️  Expected version: $EXPECTED_VERSION"

## 2. Remove old build artifacts
echo "🧹 Cleaning old builds..."
rm -rf dist/ build/ *.egg-info src/*.egg-info

## 3. Build the wheel
echo "📦 Building the wheel..."
python -m build
WHEEL_FILE=$(ls dist/*.whl)

## 4. Create an isolated test environment
ENV_NAME="test_env_fast_acbf"
echo "🧪 Creating an isolated test environment ($ENV_NAME)..."
conda create -n "$ENV_NAME" python=3.12 -y
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

## 5. Install the built wheel
echo "⚙️ Installing the fresh wheel..."
pip install "$WHEEL_FILE"

## 6. Verify import and version
echo "🔍 Verifying import and version..."
ACTUAL_VERSION=$(python -c "import fast_acbf; print(fast_acbf.__version__)")
if [ "$ACTUAL_VERSION" == "$EXPECTED_VERSION" ]; then
    echo "✅ SUCCESS: fast_acbf $ACTUAL_VERSION imported correctly."
else
    echo "❌ ERROR: version mismatch (expected $EXPECTED_VERSION, got $ACTUAL_VERSION)."
    conda deactivate
    conda remove -n "$ENV_NAME" --all -y
    exit 1
fi

## 7. Sanity-check the core public API surface is importable
echo "🔍 Verifying public API imports..."
python -c "from fast_acbf import BFSolver, Dataset4D, QualityMetrics, BFExtractor, PipelineManager"
echo "✅ SUCCESS: core public API imported correctly."

## 8. Clean up the test environment
echo "🧹 Cleaning up..."
conda deactivate
conda remove -n "$ENV_NAME" --all -y

echo "🎉 All checks passed! fast-acbf is packageable and installable."
