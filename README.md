# learnPytorch

用于系统学习 PyTorch 的实验仓库。目标是把核心概念、最小示例、训练实验和复盘笔记持续沉淀下来。

## 环境初始化

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name learn-pytorch --display-name "Python (learn-pytorch)"
```

`requirements.txt` 使用通用 pip 依赖。CUDA 用户建议按 PyTorch 官网针对本机 CUDA 版本生成的安装命令调整 `torch`、`torchvision` 和 `torchaudio`。

## 目录结构

```text
.
├── docs/            # 学习计划、阶段笔记和复盘文档
├── notebooks/       # 探索式学习笔记和实验
├── src/             # 可复用代码
├── tests/           # 测试
├── data/
│   ├── raw/         # 原始数据，不提交真实数据文件
│   └── processed/   # 处理中间数据，不提交真实数据文件
├── models/          # 模型权重，不提交真实权重文件
└── outputs/         # 图表、日志、实验输出，不提交运行产物
```

## 建议学习路线

1. Tensor 基础：形状、广播、索引、矩阵运算和设备切换。
2. Autograd：计算图、反向传播、梯度清零和 `detach`。
3. `nn.Module`：参数、层、前向传播和模型组合。
4. 数据管道：`Dataset`、`DataLoader`、transform 和 batch 形状。
5. 训练循环：loss、optimizer、评估、checkpoint 和日志。
6. 典型任务：MLP、CNN、RNN/Transformer、迁移学习和推理部署。

## 实验记录建议

- 在 notebook 或 Markdown 中记录实验目标、数据来源、关键超参数、随机种子和结论。
- 大文件只记录来源和生成方式，不直接提交到 git。
- 可复用逻辑沉淀到 `src/`，再用 notebook 调用，避免重要代码只存在于笔记中。

## 当前学习内容

- 学习计划：[docs/learning-plan.md](docs/learning-plan.md)
- 第 1 阶段 notebook：[notebooks/01_tensor_basics.ipynb](notebooks/01_tensor_basics.ipynb)
- 第 1 阶段速查笔记：[docs/notes/01_tensor-shape-cheatsheet.md](docs/notes/01_tensor-shape-cheatsheet.md)
- 第 2 阶段 notebook：[notebooks/02_autograd_linear_regression.ipynb](notebooks/02_autograd_linear_regression.ipynb)
- 第 2 阶段速查笔记：[docs/notes/02_autograd-linear-regression.md](docs/notes/02_autograd-linear-regression.md)
