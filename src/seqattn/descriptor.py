import numpy as np
import torch
from attrs import define, field
from itertools import islice

from .cache import PagedKVCacheManager, FlashInferPackedData


@define(slots=True)
class SequenceData:
    """Data structure for sequence allocation information."""

    kv_page_indices: list[int]
    last_page_len: int


@define(slots=True)
class CacheDescriptor:
    """Helper class for managing sequence-to-page mappings.

    This class provides simple sequence-level management by coordinating
    with a PagedKVCacheManager for page allocation. It handles the calculation
    of page requirements and maintains sequence metadata.
    """

    cache_manager: PagedKVCacheManager
    # Active sequence data
    seq_id_to_data: dict[int, SequenceData] = field(factory=dict)
    _page_size: int = field(init=False)

    def __attrs_post_init__(self):
        self._page_size = self.cache_manager.page_size

    def allocate_decoding(self, seq_ids: list[int]):
        """
        Allocate pages for decoding sequences (one new token).
        Args:
            seq_ids: List of decoding sequence IDs. They must be registered.
        """
        total_pages_needed = 0
        seq_page_requirements = []
        for seq_id in seq_ids:
            seq_data = self.seq_id_to_data[seq_id]
            seq_page = int((seq_data.last_page_len + 1) > self._page_size)
            seq_data.last_page_len = seq_data.last_page_len % self._page_size + 1
            # (lpl -1 +1) %ps + 1
            total_pages_needed += seq_page
            seq_page_requirements.append(seq_page)
        available_pages = self.cache_manager.allocate(total_pages_needed)
        if not available_pages:
            return
        page_offset = 0
        for seq_id, page_needed in zip(seq_ids, seq_page_requirements, strict=True):
            if page_needed > 0:
                self.seq_id_to_data[seq_id].kv_page_indices.extend(
                    islice(available_pages, page_offset, page_offset + page_needed)
                )
            page_offset += page_needed

    def allocate(self, seq_ids: list[int], seq_new_lens: list[int]):
        """Allocate pages for sequences based on their new lengths.
        Stateful. New lengths would be assumed to be appended to the end of the sequence.

        Args:
            seq_ids: List of sequence IDs
            seq_new_lens: List of sequence new lengths
        """
        if not seq_ids:
            return []

        # Calculate total pages needed for all sequences
        total_pages_needed = 0
        seq_page_requirements = []
        last_page_lens = []

        for seq_id, seq_new_len in zip(seq_ids, seq_new_lens, strict=True):
            if seq_id in self.seq_id_to_data:
                seq_len = self.seq_id_to_data[seq_id].last_page_len + seq_new_len
                page_used = 1
            else:
                seq_len = seq_new_len
                page_used = 0
            pages_needed = (
                seq_len + self._page_size - 1
            ) // self._page_size - page_used
            seq_page_requirements.append(pages_needed)
            last_page_lens.append(
                (seq_len - 1) % self._page_size + 1
            )  ##make 0 -> page_size
            total_pages_needed += pages_needed

        # Get available pages from manager
        available_pages = self.cache_manager.allocate(total_pages_needed)

        # Distribute pages to sequences
        page_offset = 0

        for seq_id, pages_needed, last_page_len in zip(
            seq_ids, seq_page_requirements, last_page_lens, strict=True
        ):
            seq_pages = available_pages[page_offset : page_offset + pages_needed]
            page_offset += pages_needed

            # Store sequence data
            if seq_id in self.seq_id_to_data:
                self.seq_id_to_data[seq_id].kv_page_indices.extend(seq_pages)
                self.seq_id_to_data[seq_id].last_page_len = last_page_len
            else:
                self.seq_id_to_data[seq_id] = SequenceData(
                    kv_page_indices=seq_pages,
                    last_page_len=last_page_len,
                )

    def release(self, seq_ids: list[int]) -> None:
        """Release sequences and their allocated pages.

        Args:
            seq_ids: List of sequence IDs to release
        """
        if not seq_ids:
            return

        # Collect all page indices to release
        all_page_indices = []
        for seq_id in seq_ids:
            seq_data = self.seq_id_to_data.get(seq_id, None)
            if seq_data is not None:
                all_page_indices.extend(seq_data.kv_page_indices)
        if all_page_indices:
            self.cache_manager.release_pages(all_page_indices)

        # Remove from active sequences dict
        for seq_id in seq_ids:
            self.seq_id_to_data.pop(seq_id, None)

    def pack_for_flashinfer(self, seq_ids: list[int]) -> FlashInferPackedData:
        """Pack sequence data for flashinfer computation format.

        Args:
            seq_ids: List of sequence IDs to pack

        Returns:
            FlashInferPackedData containing flashinfer-required tensors
        """
        kv_page_indices_list = []
        kv_page_indptr = [0]
        kv_last_page_lens = []

        for seq_id in seq_ids:
            if seq_id not in self.seq_id_to_data:
                raise ValueError(f"Sequence ID {seq_id} not found in cache descriptor")

            seq_data = self.seq_id_to_data[seq_id]
            page_indices = seq_data.kv_page_indices
            last_page_len = seq_data.last_page_len

            kv_page_indices_list.extend(page_indices)
            kv_page_indptr.append(kv_page_indptr[-1] + len(page_indices))
            kv_last_page_lens.append(last_page_len)
        return FlashInferPackedData(
            kv_page_indices=torch.tensor(kv_page_indices_list, dtype=torch.int32),
            kv_page_indptr=torch.tensor(kv_page_indptr, dtype=torch.int32),
            kv_last_page_len=torch.tensor(kv_last_page_lens, dtype=torch.int32),
        )


@define
class HeadIDGenerator:
    num_kv_heads: int
    _head_power: int = field(init=False)

    def __attrs_post_init__(self):
        self._head_power = int(np.log2(self.num_kv_heads))

    def get_head_id(self, seq_id: int, head_idx: int) -> int:
        return (seq_id << self._head_power) | head_idx 