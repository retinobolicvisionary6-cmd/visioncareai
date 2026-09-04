"""
Fundus Image Quality Assessment package.

Public entry point:
    from src.quality import assess_quality
"""

from .quality import assess_quality, result_to_json

__all__ = ["assess_quality", "result_to_json"]
