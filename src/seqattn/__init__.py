"""
seqattn - A convenient lib for sequence-level attention abstraction, powered by flashinfer.
"""

from .cache import NoPageError, FlashInferPackedData, PagedKVCacheManager
from .descriptor import SequenceData, CacheDescriptor, HeadIDGenerator

__all__ = [
    "NoPageError",
    "FlashInferPackedData", 
    "PagedKVCacheManager",
    "SequenceData",
    "CacheDescriptor",
    "HeadIDGenerator",
]

__version__ = "0.0.2"
