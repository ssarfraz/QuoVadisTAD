"""
# Time-Series Anomaly Detection (TAD)
This is the complementary software package for the paper **TODO add paper name and link** .
It contains the implementations used to generate the results in the paper.
**TODO add more details**

.. include:: ../../README.md
"""
import os
from pathlib import Path

DEFAULT_RESOURCE_DIR = Path(
	str(os.path.dirname(os.path.abspath(__file__))) + "/../../resources"
)
"""the default resource directory"""

DEFAULT_DATA_DIR = DEFAULT_RESOURCE_DIR / "processed_datasets"
"""the default data directory"""
