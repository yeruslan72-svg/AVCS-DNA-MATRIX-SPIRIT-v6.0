"""
AVCS DNA-MATRIX SPIRIT v7.0
Enhanced Data Manager Module
----------------------------
Advanced data management with caching, synchronization, 
real-time streaming, and industrial data patterns.
"""

import json
import os
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Union
from datetime import datetime, timedelta
import logging
from pathlib import Path
import pickle
import hashlib
from threading import Lock
import asyncio
from dataclasses import dataclass


@dataclass
class DataStats:
    """Statistics for data quality monitoring."""
    record_count: int
    data_size: int
    last_update: datetime
    data_quality: float  # 0.0 - 1.0
    missing_values: int


class DataManager:
    """Enhanced data manager for industrial AI systems."""

    def __init__(self, data_dir: str = "./data", cache_size: int = 1000):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        
        # Enhanced directory structure
        self.dirs = {
            'raw': self.data_dir / "raw",
            'processed': self.data_dir / "processed", 
            'cache': self.data_dir / "cache",
            'backup': self.data_dir / "backup",
            'models': self.data_dir / "models"
        }
        
        for dir_path in self.dirs.values():
            dir_path.mkdir(exist_ok=True)
        
        # Caching system
        self._cache: Dict[str, Any] = {}
        self._cache_size = cache_size
        self._cache_lock = Lock()
        
        # Real-time data buffers
        self._stream_buffers: Dict[str, List] = {}
        self._buffer_max_size = 10000
        
        # Statistics and monitoring
        self.stats: Dict[str, DataStats] = {}
        self.logger = logging.getLogger("DataManager")
        
        self.logger.info(f"DataManager initialized with base directory: {self.data_dir}")

    def _get_file_path(self, filename: str, category: str = 'processed') -> Path:
        """Get full file path with category-based organization."""
        return self.dirs[category] / filename

    def load_json(self, filename: str, category: str = 'processed', 
                  use_cache: bool = True) -> Dict[str, Any]:
        """Enhanced JSON loading with caching and validation."""
        cache_key = f"json_{category}_{filename}"
        
        # Check cache first
        if use_cache and cache_key in self._cache:
            self.logger.debug(f"Cache hit for {cache_key}")
            return self._cache[cache_key]
        
        path = self._get_file_path(filename, category)
        
        if not path.exists():
            self.logger.error(f"JSON file not found: {path}")
            raise FileNotFoundError(f"{path} not found.")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            
            # Update cache
            if use_cache:
                self._update_cache(cache_key, data)
            
            # Update statistics
            self._update_stats(str(path), data)
            
            self.logger.info(f"Loaded JSON: {path} ({len(data)} records)")
            return data
            
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            self.logger.error(f"Error loading JSON {path}: {e}")
            raise

    def save_json(self, filename: str, data: Dict[str, Any], 
                  category: str = 'processed', backup: bool = True):
        """Enhanced JSON saving with backup and compression."""
        path = self._get_file_path(filename, category)
        
        # Create backup if requested and file exists
        if backup and path.exists():
            backup_path = self.dirs['backup'] / f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            path.rename(backup_path)
            self.logger.info(f"Created backup: {backup_path}")
        
        try:
            with open(path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            # Update cache
            cache_key = f"json_{category}_{filename}"
            self._update_cache(cache_key, data)
            
            # Update statistics
            self._update_stats(str(path), data)
            
            self.logger.info(f"Saved JSON: {path} ({len(data) if isinstance(data, dict) else 'N/A'} items)")
            
        except Exception as e:
            self.logger.error(f"Error saving JSON {path}: {e}")
            raise

    def load_csv(self, filename: str, category: str = 'raw', 
                 **kwargs) -> pd.DataFrame:
        """Enhanced CSV loading with data validation."""
        cache_key = f"csv_{category}_{filename}"
        
        # Check cache
        if cache_key in self._cache:
            self.logger.debug(f"Cache hit for {cache_key}")
            return self._cache[cache_key].copy()
        
        path = self._get_file_path(filename, category)
        
        if not path.exists():
            self.logger.error(f"CSV file not found: {path}")
            raise FileNotFoundError(f"{path} not found.")
        
        try:
            # Enhanced CSV reading with common industrial data patterns
            df = pd.read_csv(
                path,
                parse_dates=True,
                infer_datetime_format=True,
                low_memory=False,
                **kwargs
            )
            
            # Data quality checks
            self._validate_dataframe(df, filename)
            
            # Update cache
            self._update_cache(cache_key, df)
            
            # Update statistics
            self._update_stats(str(path), df)
            
            self.logger.info(f"Loaded CSV: {path} ({len(df)} rows, {len(df.columns)} cols)")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV {path}: {e}")
            raise

    def save_csv(self, filename: str, df: pd.DataFrame, 
                 category: str = 'processed', index: bool = False,
                 backup: bool = True):
        """Enhanced CSV saving with data validation."""
        path = self._get_file_path(filename, category)
        
        # Backup existing file
        if backup and path.exists():
            backup_path = self.dirs['backup'] / f"{filename}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            path.rename(backup_path)
            self.logger.info(f"Created backup: {backup_path}")
        
        try:
            # Validate before saving
            self._validate_dataframe(df, filename)
            
            # Save with optimized settings for industrial data
            df.to_csv(path, index=index, encoding='utf-8')
            
            # Update cache
            cache_key = f"csv_{category}_{filename}"
            self._update_cache(cache_key, df)
            
            # Update statistics
            self._update_stats(str(path), df)
            
            self.logger.info(f"Saved CSV: {path} ({len(df)} rows, {len(df.columns)} cols)")
            
        except Exception as e:
            self.logger.error(f"Error saving CSV {path}: {e}")
            raise

    def save_model(self, model: Any, filename: str, metadata: Dict[str, Any] = None):
        """Save ML model with metadata."""
        path = self.dirs['models'] / filename
        
        model_data = {
            'model': model,
            'metadata': metadata or {},
            'saved_at': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        try:
            with open(path, 'wb') as f:
                pickle.dump(model_data, f)
            
            self.logger.info(f"Saved model: {path}")
            
        except Exception as e:
            self.logger.error(f"Error saving model {path}: {e}")
            raise

    def load_model(self, filename: str) -> Dict[str, Any]:
        """Load ML model with metadata."""
        path = self.dirs['models'] / filename
        
        if not path.exists():
            raise FileNotFoundError(f"Model file not found: {path}")
        
        try:
            with open(path, 'rb') as f:
                model_data = pickle.load(f)
            
            self.logger.info(f"Loaded model: {path}")
            return model_data
            
        except Exception as e:
            self.logger.error(f"Error loading model {path}: {e}")
            raise

    def stream_data(self, buffer_name: str, data: Any, max_size: int = None):
        """Add data to real-time stream buffer."""
        if buffer_name not in self._stream_buffers:
            self._stream_buffers[buffer_name] = []
        
        buffer = self._stream_buffers[buffer_name]
        buffer_max_size = max_size or self._buffer_max_size
        
        buffer.append({
            'timestamp': datetime.now(),
            'data': data
        })
        
        # Maintain buffer size
        if len(buffer) > buffer_max_size:
            self._stream_buffers[buffer_name] = buffer[-buffer_max_size:]

    def get_stream_data(self, buffer_name: str, last_n: int = None) -> List[Any]:
        """Get data from real-time stream buffer."""
        if buffer_name not in self._stream_buffers:
            return []
        
        buffer = self._stream_buffers[buffer_name]
        
        if last_n:
            return buffer[-last_n:]
        else:
            return buffer.copy()

    def _update_cache(self, key: str, value: Any):
        """Update cache with LRU-like behavior."""
        with self._cache_lock:
            if key in self._cache:
                # Move to end (most recently used)
                self._cache.pop(key)
            
            self._cache[key] = value
            
            # Enforce cache size limit
            if len(self._cache) > self._cache_size:
                # Remove oldest item (first inserted)
                oldest_key = next(iter(self._cache))
                self._cache.pop(oldest_key)

    def _validate_dataframe(self, df: pd.DataFrame, source: str):
        """Validate dataframe for data quality."""
        if df.empty:
            self.logger.warning(f"Empty dataframe from {source}")
            return
        
        # Check for excessive missing values
        missing_ratio = df.isnull().sum().sum() / (len(df) * len(df.columns))
        if missing_ratio > 0.3:  # 30% threshold
            self.logger.warning(f"High missing data ratio in {source}: {missing_ratio:.2%}")
        
        # Check for constant columns
        constant_cols = [col for col in df.columns if df[col].nunique() <= 1]
        if constant_cols:
            self.logger.warning(f"Constant columns in {source}: {constant_cols}")

    def _update_stats(self, file_path: str, data: Any):
        """Update file statistics."""
        if isinstance(data, pd.DataFrame):
            record_count = len(data)
            data_size = data.memory_usage(deep=True).sum()
            missing_values = data.isnull().sum().sum()
        elif isinstance(data, dict):
            record_count = len(data)
            data_size = len(json.dumps(data).encode('utf-8'))
            missing_values = 0  # Simplified for dicts
        else:
            return
        
        data_quality = 1.0 - (missing_values / (record_count * (len(data.columns) if hasattr(data, 'columns') else 1)))
        
        self.stats[file_path] = DataStats(
            record_count=record_count,
            data_size=data_size,
            last_update=datetime.now(),
            data_quality=data_quality,
            missing_values=missing_values
        )

    def get_data_quality_report(self) -> Dict[str, Any]:
        """Generate comprehensive data quality report."""
        report = {
            'generated_at': datetime.now().isoformat(),
            'total_files': len(self.stats),
            'total_records': sum(stats.record_count for stats in self.stats.values()),
            'total_size_mb': sum(stats.data_size for stats in self.stats.values()) / (1024 * 1024),
            'avg_data_quality': np.mean([stats.data_quality for stats in self.stats.values()]),
            'file_details': {}
        }
        
        for file_path, stats in self.stats.items():
            report['file_details'][file_path] = {
                'record_count': stats.record_count,
                'data_size_mb': stats.data_size / (1024 * 1024),
                'last_update': stats.last_update.isoformat(),
                'data_quality': stats.data_quality,
                'missing_values': stats.missing_values
            }
        
        return report

    def cleanup_old_backups(self, max_age_days: int = 30):
        """Remove backup files older than specified days."""
        cutoff_time = datetime.now() - timedelta(days=max_age_days)
        backup_files = list(self.dirs['backup'].glob("*.backup_*"))
        
        deleted_count = 0
        for backup_file in backup_files:
            if backup_file.stat().st_mtime < cutoff_time.timestamp():
                backup_file.unlink()
                deleted_count += 1
        
        self.logger.info(f"Cleaned up {deleted_count} old backup files")

    def list_files(self, category: str = None, pattern: str = "*") -> List[str]:
        """Enhanced file listing with filtering."""
        if category:
            base_dir = self.dirs[category]
        else:
            base_dir = self.data_dir
        
        files = []
        for file_path in base_dir.rglob(pattern):
            if file_path.is_file():
                # Return relative path from data directory
                rel_path = file_path.relative_to(self.data_dir)
                files.append(str(rel_path))
        
        return sorted(files)

    def get_file_info(self, filename: str, category: str = 'processed') -> Dict[str, Any]:
        """Get detailed information about a file."""
        path = self._get_file_path(filename, category)
        
        if not path.exists():
            return {'error': 'File not found'}
        
        stat = path.stat()
        
        return {
            'filename': filename,
            'category': category,
            'full_path': str(path),
            'size_bytes': stat.st_size,
            'size_mb': stat.st_size / (1024 * 1024),
            'created': datetime.fromtimestamp(stat.st_ctime).isoformat(),
            'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
            'file_type': path.suffix
        }
