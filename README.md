# seqattn

[中文版 README](README_zh.md)

A lightweight sequence-level attention abstraction library powered by [flashinfer](https://github.com/flashinfer-ai/flashinfer).

## Overview

seqattn provides a minimal yet powerful wrapper around flashinfer's paged attention functionality, designed with the KISS (Keep It Simple, Stupid) principle. Instead of introducing new complex concepts, it offers clean abstractions for managing sequence-level KV cache operations.

## Key Features

- **Lightweight**: Minimal overhead with clean, focused API
- **Sequence-level abstraction**: Manage attention at the sequence level rather than token level
- **Paged KV cache**: Efficient memory management with page-based allocation
- **Reference counting**: Safe memory sharing for prefix caching scenarios
- **Head-wise operations**: Support for head-wise paged attention patterns
- **flashinfer integration**: Built on top of the high-performance flashinfer library

## Core Components

### PagedKVCacheManager
Physical memory manager that handles:
- Page allocation and deallocation
- Reference counting for memory sharing
- Key-value cache storage with configurable layouts (NHD/HND)
- Direct integration with flashinfer's append operations

### CacheDescriptor
Sequence-level coordinator that provides:
- Mapping from sequence IDs to page allocations
- Automatic page requirement calculation
- Batch operations for multiple sequences
- Packaging data for flashinfer consumption

### FlashInferPackedData
Data structure containing all tensors required by flashinfer:
- Page indices and pointers
- Last page lengths for each sequence
- Device transfer utilities

## Installation

```bash
pip install seqattn
```

### FlashInfer Installation

**Important**: FlashInfer has complex distribution requirements and is not included as a direct dependency due to:

1. **PyTorch/CUDA Version Compatibility**: FlashInfer requires specific PyTorch and CUDA version combinations
2. **Multiple Installation Channels**: Different installation methods for different environments
3. **Hardware Requirements**: Only supports specific GPU architectures (`sm75`, `sm80`, `sm86`, `sm89`, `sm90`)

For more information, I strongly encourage you to check the [installation page](https://docs.flashinfer.ai/installation.html) of flashinfer.

Please install FlashInfer separately according to your environment:

**Option 1 - Prebuilt wheels (Recommended):**
```bash
# For PyTorch 2.6 + CUDA 12.6
pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/

# For other combinations, see: https://docs.flashinfer.ai/installation.html
```

**Option 2 - JIT version from PyPI:**
```bash
pip install flashinfer-python
```

**Option 3 - From source:**
```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
pip install --no-build-isolation --verbose .
```

Check your PyTorch CUDA version with:
```bash
python -c "import torch; print(torch.version.cuda)"
```

## Quick Start

```python
import torch
from seqattn import PagedKVCacheManager, CacheDescriptor

# Initialize cache manager
cache_manager = PagedKVCacheManager(
    num_pages=1024,
    page_size=16,
    num_heads=32,
    head_dim=128,
    dtype=torch.float16,
    device=torch.cuda.current_device()
)

# Create sequence descriptor
descriptor = CacheDescriptor(cache_manager)

# Allocate for sequences
seq_ids = [1, 2, 3]
seq_lengths = [100, 150, 80]
descriptor.allocate(seq_ids, seq_lengths)

# Pack for flashinfer
flashinfer_data = descriptor.pack_for_flashinfer(seq_ids)

# Use with your attention computation...
```

## Advanced Usage

### Reference Counting for Prefix Caching

```python
# Share pages between sequences with common prefixes
shared_pages = [0, 1, 2]  # Pages containing shared prefix
cache_manager.ref(shared_pages)  # Increment reference count

# Multiple sequences can now safely reference these pages
```

### Head-wise Operations

```python
from seqattn import HeadIDGenerator

# Generate unique head IDs for head-wise attention
head_gen = HeadIDGenerator(num_kv_heads=32)
head_id = head_gen.get_head_id(seq_id=1, head_idx=5)
## use head-ids as if they are seq-ids.
```

## API Reference

### PagedKVCacheManager

- `allocate(num_pages)`: Allocate pages and return indices
- `ref(page_indices)`: Increment reference count for pages
- `unref(page_indices)`: Decrement reference count
- `release_pages(page_indices)`: Release pages when ref count reaches zero
- `append_kv(keys, values, flashinfer_data, append_indptr_cpu)`: Append KV pairs

### CacheDescriptor

- `allocate(seq_ids, seq_new_lens)`: Allocate pages for sequences
- `allocate_decoding(seq_ids)`: Allocate for single-token decoding
- `release(seq_ids)`: Release sequences and their pages
- `pack_for_flashinfer(seq_ids)`: Pack data for flashinfer consumption

## Requirements

- Python >= 3.10
- torch
- numpy
- attrs
- flashinfer (install separately)

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
