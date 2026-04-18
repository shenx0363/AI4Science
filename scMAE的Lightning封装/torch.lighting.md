PyTorch Lightning 是一个基于 PyTorch 的**高级深度学习框架**，旨在将科研代码的灵活性与工程化最佳实践结合，通过标准化训练流程大幅减少模板代码。

# 用法一：将神经网络模型代码放在一个类中
```python
class BasicLightingTrain(L.LightningModule):
	def __init__(self):
		"""
		__init__()方法包含神经网络的权重、偏差、学习率
		"""
	def forward(self,input):
		"""
		forward()方法包含神经网络运行数据的连接层、激励函数等
		"""
	def configure_optimizers(self):
		"""
		包含优化神经网络的方法，比如SGD随机梯度下降
		"""
	def training_step(self,batch,batch_idx):
		"""
		该方法从我们创建的dataloader中获取一个批次的训练数据和该批次的索引，然后计算损失，return loss
		”“”
```


# 用法二：自动寻找到合适的学习率，不用手调
#### 找到合适的学习率
在pytorch包装的模型中，learning_rate是需要手动调的，而在pytorch.lighting的包装下，learning_rate可以通过训练自主找到最合适的值。

我们创建一个Lighting Trainer对象，用它来找到一个好的学习率值，然后我们将使用它来优化或训练模型
```python
model= BasicLightingTrain()
trainer=L.Trainer(max_epochs=34)
```
我们设置34个轮次来拟合数据，如果不够，Lighting可以在原地添加额外的轮次

`Trainer` 过后，通过调用 `tuner.lr_find()` 找到一个改进的学习率
```python
lr_find_results=trainer.tuner.lr_find(model,
								train_dataloders=dataloader,
								min_lr=0.001,
								max_lr=1.0,
								early_stop_threshold=None)
```
`lr_find()`需要传递的参数：神经网络模型，训练数据(dataloader)，最小学习率，最大学习率，early_stop_threshold=None告诉它不要过早停止。

默认情况下，`lr_find()`会在最小最大值之间创建100个候选学习率，通过将`early_stop_threshold=None` 测试 所有的候选学习率

通过对结果调用`suggestion()`来访问改进的学习率
将模型中的learning_rate变量设置为新的学习率
```python
new_lr=lr_find_results.suggestion()
model.learning_rate=new_lr
```

#### 将学习率用于训练，优化final_bias
在使用Lighting之前，我们通常需要大量代码用来优化b_final：
**$b_final$ : 模型的参数 $w$ 和 $b$**
我们前向传播，计算损失，对loss进行梯度下降(loss是想要优化的参数)，对loss反向传播，最后用 `optimizer.step()` 向最有参数值迈出一小步，用`optimier.zero_grad()` 将梯度归零，进行下一步轮次
**`optimizer.step()`：** 这是**最直接**操作 $b$ 的时刻。它根据刚才计算的梯度，更新 $b$ 的值：

$$b_{new} = b_{old} - \eta \cdot \nabla b$$

<img width="2285" height="1403" alt="image" src="https://github.com/user-attachments/assets/ed5f21e9-9fd3-414a-8e39-454e51e31c70" />


**使用训练器调用fit()函数**，来优化b_final：
fit()参数：模型、训练数据
```python
trainer.fit(model, train_dataloaders=dataloader)
```
训练器会调用我们自定义模型的`configure_optimizers()`函数、`training_step()`函数

# 用法三：不用大改代码就可以用GPU训练
```python
trainer=L.Trainer(max_epoches=34,accelerator="auto",devices="auto")
```
`accelerator="auto"` accelerator设置为auto让Lighting自动检测GPU是否可用，`devices="auto"` 让Lighting确定由多少个GPU可用
`accelerator="gpu"`


验证集[[验证集validation dataset]]

# 一些内置属性和函数
### `self.hparams` 和 `self.save_hyperparameters()`

`self.hparams`是**统一管理超参数**的内置属性，必须配合 `self.save_hyperparameters()` 使用 ——**不调用这个方法，self.hparams 为空**

- 作用：二者用来存取超参数、自动写入检查点、保证训练可复现、方便日志 / 可视化

- 保存：调用 `save_hyperparameters()` 后，超参数自动写入 `.ckpt` 的 `hyper_parameters` 字段Lightning AI


# `LightningDataModule` 接口

在 PyTorch Lightning 中，`LightningDataModule` 的核心设计哲学是将**数据集的处理逻辑**（下载、清理、分割、加载）从**模型训练逻辑**（梯度下降、损失计算）中完全剥离出来。

一个完整的 `DataModule` 通常包含以下五个核心钩子（Hooks）：

|**函数**|**触发时机**|**主要用途**|
|---|---|---|
|`__init__`|实例化类时|存储超参数（Batch Size, 路径, 归一化参数等）。|
|`prepare_data`|主进程执行（仅一次）|用于**下载数据**或**写入磁盘**。不要在这里给 `self` 赋值属性。|
|**`setup(stage)`**|**每个 GPU 进程执行**|执行数据分割（Train/Val/Test）和状态初始化（如加载内存、拟合归一化器）。|
|**`train_dataloader`**|训练循环开始前|返回用于训练的 `DataLoader` 实例。|
|`val_dataloader`|验证循环开始前|返回用于验证的 `DataLoader` 实例。|

train_dataloader:
**工程解耦**：你无需在 `Trainer` 外部手动处理 `DataLoader`。只需将 `datamodule` 传给 `trainer.fit()`，Lightning 会自动调用这个函数。

### DataLoader()函数

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,              # 加载的数据集 (Dataset对象)
    batch_size=1,         # 每个批次加载的样本数
    shuffle=False,        # 是否在每个 Epoch 开始时打乱数据
    num_workers=0,        # 使用多少个子进程来加载数据
    drop_last=False,      # 如果样本总数不能被 batch_size 整除，是否丢弃最后一批
    pin_memory=False,     # 是否将数据锁定在内存中（加速 GPU 传输）
    collate_fn=None       # 如何将样本列表组合成一个 Batch 的函数
)
```

# ModelCheckpoint接口
ModelCheckpoint回调能自动保存训练过程中的模型检查点，智能地根据你关注的指标（比如val_loss或val_acc）保存表现最好的模型版本。

#### 监控指标与保存策略
`monitor`参数是ModelCheckpoint的灵魂所在。它决定了回调要根据哪个指标来判断模型的好坏。这个指标必须是你通过`self.log()`或`self.log_dict()`记录过的

```python
# 在你的LightningModule中
def validation_step(self, batch, batch_idx):
    loss = ...
    acc = ...
    self.log('val_loss', loss)  # 可以被monitor监控
    self.log('val_acc', acc)    # 也可以被监控
```
设置监控指标时，`mode`参数必须与之匹配：

- 对于val_loss这种越小越好的指标，设置`mode='min'`
- 对于val_acc这种越大越好的指标，设置`mode='max'`

#### 保存数量与路径配置

`save_top_k`参数控制保存多少个最佳模型。这里有几个实用技巧：

- 设为1时只保存当前最佳（覆盖之前的）
- 设为3会保留前三名的模型
- 设为-1会保存所有检查点（慎用，会占用大量磁盘空间）

`dirpath`和`filename`参数决定了模型的保存位置和命名格式。

#### 分布式训练注意事项

在多GPU训练时，有几个关键点：

1. 确保所有进程都能访问保存路径
2. 使用`save_last=True`保证最后能保存完整模型
3. 文件名中加入rank信息避免冲突：

```python
ModelCheckpoint(
    filename='rank={rank}-{epoch}',
    ...
)
```
中文资料：
[PyTorch Lightning ModelCheckpoint实战：如何高效保存与恢复最佳模型-CSDN博客](https://blog.csdn.net/weixin_42662605/article/details/160139891)

