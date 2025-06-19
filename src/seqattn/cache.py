import numpy as np
import torch
from attrs import define, field
from collections import deque
import flashinfer
from collections import Counter

nint32 = np.dtype[np.int32]


class NoPageError(Exception):
    """Exception raised when there are no available logical pages for allocation."""

    pass


@define(slots=True)
class FlashInferPackedData:
    """Packed data structure for flashinfer computation.

    Contains all necessary tensors required by flashinfer attention and append operations.
    This structure can be reused across multiple computations to avoid redundant packing.
    """

    kv_page_indices: torch.Tensor  # [total_pages] flattened page indices
    kv_page_indptr: torch.Tensor  # [batch_size + 1] page pointers per sequence
    kv_last_page_len: torch.Tensor  # [batch_size] last page length per sequence

    def to_device(
        self, device: torch.device, non_blocking: bool = True
    ) -> "FlashInferPackedData":
        """Transfer all tensors to specified device.

        Args:
            device: Target device
            non_blocking: Whether to use non-blocking transfer

        Returns:
            New FlashInferPackedData instance with tensors on target device
        """
        return FlashInferPackedData(
            kv_page_indices=self.kv_page_indices.to(device, non_blocking=non_blocking),
            kv_page_indptr=self.kv_page_indptr.to(device, non_blocking=non_blocking),
            kv_last_page_len=self.kv_last_page_len.to(
                device, non_blocking=non_blocking
            ),
        )


@define(slots=True)
class PagedKVCacheManager:
    """Physical manager for paged KV cache memory and page allocation.

    This class manages physical memory tensors and page reference counting.
    It provides low-level page allocation interface for external allocation logic.
    """

    num_pages: int
    page_size: int
    num_heads: int
    head_dim: int
    dtype: torch.dtype
    device: torch.device
    kv_layout: str = "NHD"

    # Physical memory tensors
    key_cache: torch.Tensor = field(init=False)
    value_cache: torch.Tensor = field(init=False)

    # Page allocation state
    ref_count: np.ndarray = field(
        init=False
    )  # [num_pages], reference count for each page
    free_pages: deque = field(init=False)  # Available page indices

    def __attrs_post_init__(self):
        """Initialize memory tensors and page allocation state."""
        # Initialize page allocation
        self.ref_count = np.zeros(self.num_pages, dtype=np.int32)
        self.free_pages = deque(range(self.num_pages))

        # Initialize memory tensors based on layout
        if self.kv_layout == "NHD":
            cache_shape = (
                self.num_pages,
                self.page_size,
                self.num_heads,
                self.head_dim,
            )
        elif self.kv_layout == "HND":
            cache_shape = (
                self.num_pages,
                self.num_heads,
                self.page_size,
                self.head_dim,
            )
        else:
            raise ValueError(f"Unsupported kv_layout: {self.kv_layout}")

        self.key_cache = torch.empty(cache_shape, dtype=self.dtype, device=self.device)
        self.value_cache = torch.empty(
            cache_shape, dtype=self.dtype, device=self.device
        )

    def allocate(self, num_pages: int) -> list[int]:
        """Allocate num_pages pages. Increment reference count for each page.
        Args:
            num_pages: Number of pages to allocate
        Returns:
            List of allocated page indices
        """
        if self.available_pages_count < num_pages:
            raise NoPageError(
                f"Need {num_pages} pages but only {self.available_pages_count} available"
            )
        allocated = []
        for _ in range(num_pages):
            allocated.append(self.free_pages.popleft())
            self.ref_count[allocated[-1]] += 1
        return allocated

    def ref(self, page_indices: list[int]) -> None:
        """Claim a new ref for the given page indices.
        Args:
            page_indices: List of page indices to claim
        Warning:
            Never ref a free page.
            It's slow to test whether a page is free(since it's a deque)
            so we don't do this check.
            It's enough in many cases, you mostly want to ref a page that's already allocated (like prefix caching).
        """
        for page_idx in page_indices:
            self.ref_count[page_idx] += 1

    def unref(self, page_indices: list[int]) -> None:
        for page_idx in page_indices:
            self.ref_count[page_idx] -= 1
            if self.ref_count[page_idx] == 0:
                self.free_pages.append(page_idx)

    def release_pages(self, page_indices: list[int]) -> None:
        """Release pages and decrement reference counts.

        Args:
            page_indices: List of page indices to release
        """
        if not page_indices:
            return

        # Track which pages become free to avoid duplicates
        page_counter = Counter(page_indices)

        # Directly decrement ref count for each page
        for page_idx, count in page_counter.items():
            self.ref_count[page_idx] -= count
            if self.ref_count[page_idx] == 0:
                self.free_pages.append(page_idx)

    @property
    def available_pages_count(self) -> int:
        """Get number of available pages."""
        return len(self.free_pages)

    def get_memory_tensors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Get the underlying memory tensors.

        Returns:
            Tuple of (key_cache, value_cache) tensors
        """
        return self.key_cache, self.value_cache

    def append_kv(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        flashinfer_data: FlashInferPackedData,
        append_indptr_cpu: torch.Tensor,
    ) -> None:
        """Append key-value pairs to the managed memory.

        Args:
            keys: Key tensor to append [total_len, num_kv_heads, head_dim]
            values: Value tensor to append [total_len, num_kv_heads, head_dim]
            flashinfer_data: Packed flashinfer data containing page allocation info
            append_indptr_cpu: Indptr tensor for keys/values to append [batch_size + 1] on CPU
        """
        # Calculate append lengths from indptr
        append_lengths_tensor = torch.diff(append_indptr_cpu)

        # Get batch indices and positions
        # NOTE: flashinfer quirk - only prefill wrapper prefers CPU indptr, others default to GPU
        batch_indices, positions = flashinfer.page.get_batch_indices_positions(
            append_indptr_cpu.to(keys.device, non_blocking=True),
            append_lengths_tensor.to(keys.device, non_blocking=True),
            int(append_indptr_cpu[-1].item()),  # Ensure int type for linter
        )

        # Transfer flashinfer data to target device
        flashinfer_data_gpu = flashinfer_data.to_device(keys.device)

        # Call flashinfer append function - ensure all tensors are on correct device
        flashinfer.page.append_paged_kv_cache(
            keys,
            values,
            batch_indices,
            positions,
            (self.key_cache, self.value_cache),
            flashinfer_data_gpu.kv_page_indices,
            flashinfer_data_gpu.kv_page_indptr,
            flashinfer_data_gpu.kv_last_page_len,
            kv_layout=self.kv_layout,
        ) 