"""Run Tests 1.1-13.9 and display actual-versus-expected comparisons."""

from __future__ import annotations

import os
import re
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# This must be set before unittest imports test_tests_1_to_7.py.
os.environ["SHOW_TEST_OUTPUTS"] = "1"

discovered = unittest.defaultTestLoader.discover(
    str(ROOT / "FunctionalTests"), pattern="test_*.py"
)


def individual_tests(suite: unittest.TestSuite):
    """Flatten a discovered suite so it can be ordered by test number."""

    for item in suite:
        if isinstance(item, unittest.TestSuite):
            yield from individual_tests(item)
        else:
            yield item


def test_number(test: unittest.TestCase) -> tuple[int, int]:
    match = re.search(r"\.test_(\d+)_(\d+)_", test.id())
    return (int(match.group(1)), int(match.group(2))) if match else (999, 999)


suite = unittest.TestSuite(sorted(individual_tests(discovered), key=test_number))
result = unittest.TextTestRunner(
    stream=sys.stdout, verbosity=2, buffer=False
).run(suite)
raise SystemExit(0 if result.wasSuccessful() else 1)
