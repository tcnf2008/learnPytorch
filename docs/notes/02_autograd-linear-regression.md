# Autograd 和手写线性回归速查

## 核心链路

第二阶段只关注一条最小训练链路：

```text
x, y -> y_hat = w * x + b -> loss -> loss.backward() -> w.grad / b.grad -> 手动更新参数
```

这里故意不使用 `nn.Linear` 和 optimizer，目的是看清楚 Autograd 如何从 loss 推回参数梯度。

## 计算图

当 Tensor 满足 `requires_grad=True`，并参与可求导运算时，PyTorch 会动态构建计算图。

```python
w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(-1.0, requires_grad=True)
x = torch.tensor([[1.0], [2.0]])

y_hat = w * x + b
loss = (y_hat ** 2).mean()
loss.backward()

print(w.grad)
print(b.grad)
```

第一性原理：`backward()` 做的事情就是从标量 loss 出发，沿计算图用链式法则把梯度传回叶子参数。

## 叶子 Tensor

通常手动创建且设置了 `requires_grad=True` 的参数是叶子 Tensor：

```python
w = torch.tensor(1.0, requires_grad=True)
y = w * 2

print(w.is_leaf)  # True
print(y.is_leaf)  # False
```

默认只有叶子 Tensor 的 `.grad` 会被保留。中间 Tensor 如果需要查看梯度，要调用 `retain_grad()`。

## 梯度会累积

PyTorch 不会自动清空 `.grad`。多次调用 `backward()` 会把梯度加到已有 `.grad` 上。

标准训练循环必须包含：

```python
loss.backward()

with torch.no_grad():
    w -= learning_rate * w.grad
    b -= learning_rate * b.grad
    w.grad.zero_()
    b.grad.zero_()
```

如果忘记清零梯度，参数更新方向会混入历史 batch 的梯度，训练行为会变得难以解释。

## `no_grad()`、`detach()`、冻结参数

三者解决的问题不同：

- `with torch.no_grad()`：上下文中不构建计算图，常用于参数更新和评估推理。
- `tensor.detach()`：从当前计算图切断，返回不再追踪梯度的新 Tensor，通常仍共享底层数据。
- `param.requires_grad = False`：让参数不再参与梯度计算，常用于冻结模型层。

注意：`model.eval()` 只切换模块行为，不会关闭梯度。评估阶段通常同时使用：

```python
model.eval()
with torch.no_grad():
    ...
```

## Inplace 操作

带下划线的方法通常是 inplace 操作，例如：

```python
x.add_(1)
x.mul_(2)
```

不是所有 inplace 操作都错误，但如果它修改了反向传播需要的前向中间值，Autograd 会报错。学习阶段的保守策略：

- 不要对需要求导的叶子 Tensor 直接做 inplace 修改。
- 参数更新放在 `torch.no_grad()` 里。
- 不确定时先写非 inplace 版本。

## 手写线性回归最小训练循环

```python
for epoch in range(epochs):
    y_hat = w * x + b
    loss = ((y_hat - y) ** 2).mean()
    loss.backward()

    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
        w.grad.zero_()
        b.grad.zero_()
```

检查点：

- `w` 和 `b` 必须是 `requires_grad=True` 的叶子 Tensor。
- `loss` 必须是标量 Tensor。
- 每轮都要先 `backward()`，再更新参数，再清零梯度。
- 更新参数时不要构建新的计算图。

## 训练不收敛先查什么

1. `x` 和 `y` 的 shape 是否一致。
2. `y_hat` 是否真的由 `w`、`b` 计算得到。
3. `loss` 是否用 `y_hat` 和 `y` 计算。
4. `w.grad`、`b.grad` 是否为 `None`、全 0 或异常大。
5. 是否忘记清零梯度。
6. 学习率是否过大或过小。
7. 输入数据尺度是否过大。
8. 能否先在无噪声小数据上拟合到接近真实参数。

## 第二阶段验收

完成第二阶段后，应该能够：

- 画出 `w * x + b -> loss -> backward -> grad` 的计算图。
- 解释为什么梯度会累积，为什么要清零。
- 不用 `nn.Linear` 写出一个完整线性回归训练循环。
- 区分 `no_grad()`、`detach()` 和 `requires_grad=False`。
- 解释常见 inplace 操作为什么会破坏梯度计算。
