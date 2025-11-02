"""
AVCS DNA-MATRIX SPIRIT v6.0
Data Manager Module
--------------------
Responsible for handling data flow between modules,
loading datasets, caching, and ensuring synchronization between
physical and digital layers.
"""

import json
import os
import pandas as pd
from typing import Dict, Any


class DataManager:
    """Manages data storage, retrieval, and synchronization."""

    def __init__(self, data_dir: str = "./data"):
        self.data_dir = data_dir
        os.makedirs(self.data_dir, exist_ok=True)

    def load_json(self, filename: str) -> Dict[str, Any]:
        """Load data from JSON file."""
        path = os.path.join(self.data_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found.")
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save_json(self, filename: str, data: Dict[str, Any]):
        """Save data as JSON file."""
        path = os.path.join(self.data_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)

    def load_csv(self, filename: str) -> pd.DataFrame:
        """Load CSV dataset."""
        path = os.path.join(self.data_dir, filename)
        return pd.read_csv(path)

    def save_csv(self, filename: str, df: pd.DataFrame):
        """Save dataframe as CSV."""
        path = os.path.join(self.data_dir, filename)
        df.to_csv(path, index=False)

    def list_files(self) -> list:
        """List all data files in directory."""
        return os.listdir(self.data_dir)
