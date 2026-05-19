"""Regression test for the exhaustive PipelineManager policy matrix script."""

from __future__ import annotations

import subprocess
import sys


def test_pipeline_matrix_script_passes():
    result = subprocess.run(
        [sys.executable, "scripts/check_pipeline_matrix.py", "--no-tree"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Policy invariants passed." in result.stdout
