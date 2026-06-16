# PyTorch 学习计划

## 目标

这个计划的目标不是快速浏览 PyTorch API，而是建立能独立完成模型训练、调试、复盘和沉淀的能力。

学习主线：

```text
数据 -> Tensor -> forward -> loss -> backward -> optimizer.step -> 评估 -> 保存/加载 -> 推理
```

每个阶段都遵循同一个闭环：

```text
概念理解 -> 最小实验 -> 完整训练 -> 复盘笔记 -> 可复用代码
```

## 仓库使用方式

- `notebooks/`：放探索式学习和实验记录。
- `src/`：放已经理解清楚、可以复用的代码。
- `tests/`：放针对 `src/` 代码的测试。
- `data/`：放本地数据，不提交真实数据文件。
- `models/`：放本地模型权重，不提交 checkpoint。
- `outputs/`：放本地日志、图表和实验产物。

建议命名方式：

```text
notebooks/01_tensor_basics.ipynb
notebooks/02_autograd_linear_regression.ipynb
notebooks/03_training_loop_mlp.ipynb
notebooks/04_dataset_dataloader.ipynb
notebooks/05_cnn_mnist.ipynb
notebooks/06_transformer_basics.ipynb
notebooks/07_engineering_practice.ipynb
```

## 第 1 阶段：Tensor 和形状

核心问题：

- Tensor 和 Python list、NumPy array 的本质区别是什么？
- `shape`、`dtype`、`device` 分别影响什么？
- 广播、索引、切片、矩阵乘法什么时候会改变形状？
- `cpu`、`mps`、`cuda` 之间如何安全切换？
- NumPy 和 Tensor 相互转换时，什么时候共享内存，什么时候发生拷贝？
- `view()`、`reshape()`、`permute()`、`transpose()` 和 `contiguous()` 的关系是什么？

建议实验：

- 创建不同维度的 Tensor，记录每一步的 `shape`。
- 对比 `view`、`reshape`、`permute`、`transpose`。
- 手动实现一个 batch 矩阵乘法小例子。
- 写一个 `get_device()` 辅助函数，优先使用 `cuda`，其次 `mps`，最后 `cpu`。
- 对比 `torch.from_numpy()`、`torch.as_tensor()`、`torch.tensor()` 和 `.clone()` 的内存行为。
- 故意制造一个不连续 Tensor，观察 `.view()` 的报错，再用 `.contiguous()` 或 `.reshape()` 修复。

易错点：

- `torch.from_numpy()` 和 `torch.as_tensor()` 可能共享底层内存，修改一方会影响另一方。
- `permute()` 或 `transpose()` 后的 Tensor 往往不再连续，直接 `.view()` 可能失败。
- 不要只记 API 名称，要能解释形状变化和内存布局变化。

阶段产物：

- `notebooks/01_tensor_basics.ipynb`
- `docs/notes/01_tensor-shape-cheatsheet.md`
- `src/learn_pytorch/device.py`

验收标准：

- 看到一段模型代码时，可以手动推导主要 Tensor 的输入输出形状。
- 能解释广播为什么成立，或者为什么报 shape mismatch。

## 第 2 阶段：Autograd 和最小训练

核心问题：

- 计算图是什么？
- `requires_grad`、`grad`、`backward()` 分别做什么？
- 为什么每轮训练前要清零梯度？
- `detach()`、`no_grad()`、`eval()` 分别解决什么问题？
- 叶子 Tensor 是什么？为什么默认只有叶子 Tensor 会保留 `.grad`？
- 原地操作什么时候会破坏 Autograd 所需的中间值？
- `no_grad()`、`detach()`、`requires_grad = False` 的区别是什么？

建议实验：

- 用 PyTorch 手动拟合一条直线 `y = wx + b`。
- 不使用 `nn.Linear`，先手动创建 `w` 和 `b`。
- 打印每轮的 loss、`w.grad`、`b.grad`。
- 故意去掉 `zero_grad()`，观察梯度累积带来的影响。
- 故意对需要求导的 Tensor 做不合适的 inplace 操作，观察反向传播报错。
- 对比 `with torch.no_grad()` 和 `tensor.detach()` 在计算图构建上的差异。

易错点：

- 不是所有 inplace 操作都必然错误，但如果它改掉了反向传播需要的前向中间值，就会破坏梯度计算。
- `detach()` 返回的是从计算图断开的 Tensor，可能仍共享底层数据；需要隔离数据时再配合 `.clone()`。
- `eval()` 只切换模块行为，不等于关闭梯度；评估阶段通常还要配合 `torch.no_grad()`。

阶段产物：

- `notebooks/02_autograd_linear_regression.ipynb`
- 一段手写线性回归训练代码。

验收标准：

- 能说清楚 loss 如何通过计算图影响参数更新。
- 训练不收敛时，知道先检查学习率、梯度、数据尺度和 loss。

## 第 3 阶段：nn.Module 和标准训练循环

核心问题：

- `nn.Module` 如何管理参数？
- `forward()` 和直接调用模型对象有什么关系？
- loss、optimizer、`train()`、`eval()` 的职责分别是什么？
- 一个标准训练循环最少需要哪些步骤？
- 权重初始化如何影响收敛速度和稳定性？
- 为什么 `nn.CrossEntropyLoss` 的输入应该是 logits，而不是先经过 `Softmax` 的概率？

建议实验：

- 用 `nn.Module` 重写线性回归。
- 写一个小型 MLP 完成二维点分类。
- 把训练循环拆成 `train_one_epoch()` 和 `evaluate()`。
- 把模型、loss、optimizer、metric 的职责分清楚。
- 对 MLP 尝试默认初始化、Xavier 初始化和 Kaiming 初始化，对比 loss 曲线。
- 故意在分类模型最后加 `Softmax` 再接 `CrossEntropyLoss`，观察训练表现并解释原因。

易错点：

- `CrossEntropyLoss` 内部已经包含 `LogSoftmax` 和负对数似然，模型输出原始 logits 即可。
- optimizer 只会更新传入的参数；冻结层或新增层后，要检查 optimizer 是否包含正确参数。
- loss 降不下来时，先确认模型输出、label 形状和 label dtype 是否匹配。

阶段产物：

- `notebooks/03_training_loop_mlp.ipynb`
- `src/training.py`
- `src/models.py`

验收标准：

- 不看模板也能写出完整训练循环。
- 能解释 `model.train()` 和 `model.eval()` 为什么会影响 Dropout、BatchNorm 等层。

## 第 4 阶段：Dataset 和 DataLoader

核心问题：

- `Dataset` 的 `__len__` 和 `__getitem__` 应该返回什么？
- batch 维度从哪里来？
- `shuffle`、`num_workers`、`collate_fn` 分别解决什么问题？
- 数据预处理应该放在 Dataset、transform，还是训练循环里？
- macOS/Windows 上 `num_workers > 0` 为什么更容易暴露序列化和启动开销问题？
- 变长文本、变长序列或不同尺寸图像为什么不能直接 stack 成 batch？
- `pin_memory=True` 主要解决什么传输问题，为什么不必一开始就依赖它？

建议实验：

- 自定义一个小型 CSV Dataset。
- 使用 `DataLoader` 生成 batch。
- 打印单个样本和 batch 的形状。
- 写一个简单的 train/validation split。
- 编写一个自定义 `collate_fn`，把变长一维 Tensor padding 成统一长度。
- 对比 `num_workers=0` 和 `num_workers>0` 的行为，记录本机系统下的差异。

易错点：

- `__getitem__` 返回的 label 类型会影响 loss，例如分类任务通常需要 `LongTensor` 类别索引。
- 默认 `collate_fn` 要求样本可以 stack；变长数据需要 padding、截断或自定义打包逻辑。
- `pin_memory=True` 对 CUDA 主机到设备传输更有意义，学习初期先保证数据管道正确。

阶段产物：

- `notebooks/04_dataset_dataloader.ipynb`
- `src/data.py`

验收标准：

- 能把本地结构化数据整理成 PyTorch 可训练输入。
- 能定位 batch 维度、label 类型、device 不一致等常见问题。

## 第 5 阶段：CNN 和图像任务

核心问题：

- 图像 Tensor 为什么通常是 `[N, C, H, W]`？
- 卷积核、stride、padding 如何影响输出尺寸？
- pooling、flatten、linear 之间如何衔接？
- 数据增强会改变什么，不会改变什么？

建议实验：

- 用 MNIST 或 FashionMNIST 训练一个小型 CNN。
- 手动计算每层输出形状，再用代码验证。
- 对比无增强和简单增强的训练结果。
- 加载一个 `torchvision.models` 预训练模型，冻结特征提取层，只训练新的分类头。
- 保存最佳 checkpoint，并写推理函数加载模型预测单张图片。

阶段产物：

- `notebooks/05_cnn_mnist.ipynb`
- `src/vision_models.py`
- 一个迁移学习最小实验：加载预训练模型、替换分类头、冻结部分参数、完成训练和推理。

验收标准：

- 能独立设计一个小型 CNN，并正确计算进入全连接层前的维度。
- 能完成训练、评估、保存、加载和推理闭环。
- 能解释从头训练、特征提取和微调的区别。

## 第 6 阶段：序列模型和 Transformer 基础

核心问题：

- 序列任务的输入形状通常如何表示？
- RNN/LSTM 和 Self-Attention 处理序列的本质区别是什么？
- embedding 的作用是什么？
- attention 解决了什么问题？
- mask 在 Transformer 中为什么重要？
- Padding mask 和 causal mask 分别在屏蔽什么？
- Multi-head Attention 中 `[B, S, E]`、`[B, H, S, D]` 这些形状如何转换？

建议实验：

- 先写一个极简 RNN/LSTM 序列分类例子，再对比 Self-Attention 的并行计算方式。
- 从 embedding 和位置编码开始，构造一个最小序列分类例子。
- 打印 attention 输入输出形状。
- 对比 RNN 思路和 self-attention 思路。
- 构造 padding mask 和 causal mask，观察 mask 前后 attention score 的变化。
- 手动推导 Multi-head Attention 中 `view`、`transpose`、`matmul` 的形状变化。

阶段产物：

- `notebooks/06_transformer_basics.ipynb`
- 一页笔记：embedding、attention、mask 和输出形状。

验收标准：

- 能解释 Transformer 中 token、embedding、sequence length、batch size 的关系。
- 能读懂一个基础 Transformer block 的主要数据流。

## 第 7 阶段：工程实践与实验管理

核心问题：

- 一个可恢复训练的 checkpoint 应该保存哪些内容？
- 如何判断训练是正常波动、过拟合、欠拟合，还是梯度异常？
- 梯度裁剪解决什么问题，什么时候应该使用？
- TensorBoard 这类工具比 `print` 日志多解决了什么问题？
- 混合精度训练适合什么时候学，为什么不应该早于基础 FP32 训练闭环？

建议实验：

- 保存 checkpoint 时同时保存 `model.state_dict()`、`optimizer.state_dict()`、epoch、best metric 和关键配置。
- 写一个 resume 训练实验，从 checkpoint 恢复并继续训练。
- 在一个小模型上使用 `torch.nn.utils.clip_grad_norm_`，观察梯度范数变化。
- 使用 TensorBoard 记录 loss、metric、learning rate 和少量预测样例。
- 把一次实验的配置、随机种子、数据版本和结果写成复盘文档。

阶段产物：

- `notebooks/07_engineering_practice.ipynb`
- `src/checkpointing.py`
- `src/metrics.py`
- 一份实验复盘文档，记录配置、曲线、结论和下一步问题。

验收标准：

- 能从 checkpoint 恢复模型、优化器和训练进度。
- 能用曲线和指标判断训练状态，而不是只看最后一个 loss。
- 能说明梯度裁剪、TensorBoard、混合精度各自解决的问题和适用时机。

## 每个实验的记录模板

建议每个 notebook 或实验文档都包含以下内容：

```text
1. 本次实验目标
2. 数据来源和输入输出定义
3. 关键 Tensor 形状
4. 模型结构
5. loss、optimizer、metric
6. 训练配置：epoch、batch size、learning rate、seed、device
7. 实验结果
8. 出错记录和定位过程
9. 可以沉淀到 src/ 的代码
10. 下一步问题
```

## 调试清单

训练不动时，先按顺序检查：

1. 数据和标签是否对应。
2. 输入 Tensor 的 shape、dtype、device 是否正确。
3. loss 是否真的基于模型输出和 label 计算。
4. 是否调用了 `optimizer.zero_grad()`、`loss.backward()`、`optimizer.step()`。
5. 参数的梯度是否为 `None`、全 0 或异常大。
6. 学习率是否过大或过小。
7. 训练集上是否能先过拟合一个很小的 batch。
8. `train()` 和 `eval()` 模式是否使用正确。
9. 分类任务是否把 logits 直接传给了 `CrossEntropyLoss`。
10. 是否有不合适的 inplace 操作破坏了反向传播。
11. checkpoint 恢复后 optimizer 状态、epoch 和 best metric 是否同步恢复。

## 推荐节奏

如果每天有 1 到 2 小时，可以按 5 周推进：

```text
第 1 周：Tensor、shape、device、Autograd
第 2 周：nn.Module、训练循环、Dataset/DataLoader
第 3 周：MLP、CNN、保存加载、推理
第 4 周：迁移学习、序列模型、Transformer 基础
第 5 周：checkpoint、TensorBoard、实验复盘、整理 src/ 代码
```

如果时间更少，就不要压缩内容，而是拉长周期。PyTorch 的学习重点不在于看完多少材料，而在于能不能自己把数据、模型、训练和调试串起来。

## 长期沉淀标准

一个主题学完后，至少留下三类资产：

- 一个可运行 notebook。
- 一段可复用的 `src/` 代码。
- 一段复盘说明，记录原理、坑点和下一步问题。

当这些资产能够反复复用，说明这个仓库开始从“学习记录”变成“个人 PyTorch 工具箱”。
