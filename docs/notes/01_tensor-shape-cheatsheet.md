# Tensor 和形状速查

## 先看四件事

拿到任何 Tensor，先确认：

```python
tensor.shape
tensor.dtype
tensor.device
tensor.stride()
```

- `shape`：逻辑维度，决定能不能做矩阵乘法、广播、拼接。
- `dtype`：数值类型，影响精度、显存和 loss 的输入要求。
- `device`：数据在哪个设备上，模型和数据必须在同一设备。
- `stride`：内存步长，决定 Tensor 在内存里是不是连续。

## 常见形状约定

```text
标量：[]
向量：[D]
矩阵：[N, D]
表格 batch：[B, D]
图像 batch：[B, C, H, W]
序列 batch：[B, S] 或 [B, S, E]
分类 logits：[B, num_classes]
```

## 形状变换

- `reshape()`：改变逻辑形状，必要时可能复制数据。
- `view()`：改变逻辑形状，但要求底层内存布局兼容。
- `permute()`：任意交换维度，常导致 Tensor 不连续。
- `transpose()`：交换两个维度，是常见的 `permute()` 特例。
- `contiguous()`：把不连续 Tensor 复制成连续内存布局。

常见坑：

```python
x = torch.arange(24).reshape(2, 3, 4)
y = x.transpose(1, 2)
y.view(2, 12)  # 可能报错，因为 y 不连续
y.contiguous().view(2, 12)
```

## 广播规则

广播从最后一个维度开始对齐：

```text
[B, D] + [D] -> [B, D]
[B, 1] + [B, D] -> [B, D]
[B, S, D] + [D] -> [B, S, D]
```

能够广播的前提：

- 两个维度相等；或者
- 其中一个维度是 `1`；或者
- 其中一个 Tensor 没有这个维度。

## 矩阵乘法

普通二维矩阵：

```text
[N, M] @ [M, P] -> [N, P]
```

Batch 矩阵乘法：

```text
[B, N, M] @ [B, M, P] -> [B, N, P]
```

第一性原理：最后两个维度做矩阵乘法，前面的维度当作 batch 维。

## NumPy 和 Tensor 的内存关系

- `torch.from_numpy(array)`：通常共享 NumPy 底层内存。
- `torch.as_tensor(array)`：能共享时尽量共享。
- `torch.tensor(array)`：复制数据，生成新的 Tensor。
- `tensor.clone()`：复制 Tensor 数据。

如果后续还会修改原始 NumPy array，优先显式选择是否复制，避免数据被悄悄改掉。

## 设备选择

本项目使用：

```python
from learn_pytorch import get_device

device = get_device()
```

默认顺序：

```text
cuda -> mps -> cpu
```

常见坑：

- 模型和输入 Tensor 不在同一设备会报错。
- `tensor.to(device)` 返回新 Tensor，不一定原地修改。
- CPU、CUDA、MPS 的算子支持和数值表现可能有差异，学习阶段先保证逻辑正确。

## 第一阶段验收

完成第一阶段后，应该能够：

- 看到一个 Tensor 后说清楚它的 shape、dtype、device 和 stride。
- 手动推导常见 shape 变换结果。
- 解释为什么某些广播成立，某些 shape mismatch。
- 解释 NumPy 和 Tensor 转换时是否共享内存。
- 解释为什么 transpose/permute 后直接 view 可能失败。
