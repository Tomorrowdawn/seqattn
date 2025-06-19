# seqattn

[English README](README.md)

一个基于 [flashinfer](https://github.com/flashinfer-ai/flashinfer) 的轻量级序列级注意力抽象库。

## 概述

seqattn 为 flashinfer 的分页注意力功能提供了一个最小但功能强大的封装，遵循 KISS（Keep It Simple, Stupid）原则设计。它不引入新的复杂概念，而是为管理序列级 KV 缓存操作提供清晰的抽象。

## 核心特性

- **轻量级**：最小开销，API 清晰专注
- **序列级抽象**：在序列级别而非令牌级别管理注意力
- **分页 KV 缓存**：基于分页分配的高效内存管理
- **引用计数**：支持前缀缓存场景的安全内存共享
- **头级操作**：支持头级分页注意力模式
- **flashinfer 集成**：基于高性能 flashinfer 库构建

## 核心组件

### PagedKVCacheManager
物理内存管理器，负责：
- 页面分配和释放
- 内存共享的引用计数
- 支持可配置布局（NHD/HND）的键值缓存存储
- 与 flashinfer 追加操作的直接集成

### CacheDescriptor
序列级协调器，提供：
- 序列 ID 到页面分配的映射
- 自动页面需求计算
- 多序列批量操作
- 为 flashinfer 消费打包数据

### FlashInferPackedData
包含 flashinfer 所需所有张量的数据结构：
- 页面索引和指针
- 每个序列的最后页面长度
- 设备传输工具

## 安装

```bash
pip install seqattn
```

### FlashInfer 安装

**重要说明**：FlashInfer 由于复杂的分发要求而未作为直接依赖包含，主要原因包括：

1. **PyTorch/CUDA 版本兼容性**：FlashInfer 需要特定的 PyTorch 和 CUDA 版本组合
2. **多种安装渠道**：不同环境需要不同的安装方法
3. **硬件要求**：仅支持特定的 GPU 架构（`sm75`, `sm80`, `sm86`, `sm89`, `sm90`）

强烈建议您查阅flashinfer的[安装指引](https://docs.flashinfer.ai/installation.html).

请根据您的环境单独安装 FlashInfer：

**选项 1 - 预构建 wheels（推荐）：**
```bash
# 适用于 PyTorch 2.6 + CUDA 12.6
pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6/

# 其他组合请参考：https://docs.flashinfer.ai/installation.html
```

**选项 2 - 从 PyPI 安装 JIT 版本：**
```bash
pip install flashinfer-python
```

**选项 3 - 从源码安装：**
```bash
git clone https://github.com/flashinfer-ai/flashinfer.git --recursive
cd flashinfer
pip install --no-build-isolation --verbose .
```

检查您的 PyTorch CUDA 版本：
```bash
python -c "import torch; print(torch.version.cuda)"
```

## 快速开始

```python
import torch
from seqattn import PagedKVCacheManager, CacheDescriptor

# 初始化缓存管理器
cache_manager = PagedKVCacheManager(
    num_pages=1024,
    page_size=16,
    num_heads=32,
    head_dim=128,
    dtype=torch.float16,
    device=torch.cuda.current_device()
)

# 创建序列描述符
descriptor = CacheDescriptor(cache_manager)

# 为序列分配内存
seq_ids = [1, 2, 3]
seq_lengths = [100, 150, 80]
descriptor.allocate(seq_ids, seq_lengths)

# 为 flashinfer 打包数据
flashinfer_data = descriptor.pack_for_flashinfer(seq_ids)

# 在注意力计算中使用...
```

## 高级用法

### 前缀缓存的引用计数

```python
# 在具有公共前缀的序列之间共享页面
shared_pages = [0, 1, 2]  # 包含共享前缀的页面
cache_manager.ref(shared_pages)  # 增加引用计数

# 多个序列现在可以安全地引用这些页面
```

### 头级操作

```python
from seqattn import HeadIDGenerator

# 为头级注意力生成唯一的头 ID
head_gen = HeadIDGenerator(num_kv_heads=32)
head_id = head_gen.get_head_id(seq_id=1, head_idx=5)
# 然后仿佛句子一样使用这些ID.
```

## API 参考

### PagedKVCacheManager

- `allocate(num_pages)`：分配页面并返回索引
- `ref(page_indices)`：增加页面的引用计数
- `unref(page_indices)`：减少引用计数
- `release_pages(page_indices)`：当引用计数为零时释放页面
- `append_kv(keys, values, flashinfer_data, append_indptr_cpu)`：追加 KV 对

### CacheDescriptor

- `allocate(seq_ids, seq_new_lens)`：为序列分配页面
- `allocate_decoding(seq_ids)`：为单令牌解码分配
- `release(seq_ids)`：释放序列及其页面
- `pack_for_flashinfer(seq_ids)`：为 flashinfer 消费打包数据

## 依赖要求

- Python >= 3.10
- torch
- numpy
- attrs
- flashinfer

## 许可证

MIT 许可证

## 贡献

欢迎贡献！请随时提交 Pull Request。 