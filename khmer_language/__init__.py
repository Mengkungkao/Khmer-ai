"""KhmerAI - a Khmer-native AI stack built from first principles.

See README.md for the project roadmap. This package currently implements
Project 1: the Khmer Unicode & linguistic engine (`khmer_language.unicode`)
and the top-level `analyze()` entry point (`khmer_language.analyzer`).
"""

from .analyzer import analyze, format_analysis

__all__ = ["analyze", "format_analysis"]
__version__ = "0.1.0"
