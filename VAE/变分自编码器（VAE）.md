简单来说，它是一种生成式ai模型，在自编码器的基础上，用一个已经训练好的自编码器，从该潜在表示中重构图像，生成新数据。

映射：
Data Distribution   ----> Latent Distribution
数据分布                 ----> 潜在分布
 ![[Pasted image 20260410211940.png]]
Latent Distribution潜在分布代表了潜在变量在低维空间的向量捕捉数据核心的特征

需要**映射**来连接这两个不在同一空间的两个分布
1.**后验分布**：给出一个特定图像x，则生成潜在向量是z的概率 $p(z|x)$
2.**似然分布**：给出一个潜在的z，能从z重构图像x的概率 $p(x|z)$

## VAE具体过程

在自编码器中，我们把原数据压缩成几个最重要的特征，这些特征的不同组合（潜在向量）存储在一个潜在空间[二维，可能是高维，比如经过主成分分析变成二维]。但是这些潜在向量z在潜在空间里都是一个个点，所以当我们从训练好的自编码器中任意选择一个点去解码成图像，它会和原图像相差得离谱，而且潜在空间的中间位置可能甚至没有潜在向量。为了解决这种问题我们把点变成置于潜在空间中心的标准正太分布（高斯分布），即**让每个x对应的潜在分布够靠近标准正态**。

我们要的效果是从潜在空间中任意取一个潜在向量z，通过解码器还原成一个图像x'，并且这个x‘是x的**同类新样本**。于是我们希望求出原图像x与潜在向量z的后验分布p(z|x)，即给出图像x求生成的潜在向量z。

结合上面两段的描述，可知我们要训练出的潜在分布长下面这样：
$p(z|x)$ 是高斯分布
$$
p(z|x) = \mathcal{N}\left(z; \mu(x), \sigma^2(x)\right)
$$
这里的方差项时**协方差矩阵**
流程：
- 我们假设先验$p(z)=N(0,I)$（标准正态）
- 编码器输出的是**每个样本 x 对应的后验分布** $p(z∣x)$
- KL 项的作用是：**把每个$p(z|x)$ 往先验$p(z)$ 拉**，让整体潜在空间 “聚拢、连续、可采样”

然而后验分布$p(z|x)$是很难计算和训练的，所以我们需要用到数学知识**变分推断**，找一个近似分布$q$来近似真实后验分布$p$
经过**变分推断和变分下界**可知VAE的损失函数如下（数学推导这里略过）：
$$
\mathcal{L}_{VAE} = \mathcal{L}_{rec} + \mathcal{L}_{KL}
$$
**变分自编码器总损失**=**重构损失**(均方误差MSE) + **KL散度损失**(标准正态先验)
$$
\mathcal{L}_{rec} = MSE
$$
$$
\mathcal{L}_{KL} = \frac{1}{2} \sum_{i=1}^{d} \left( \sigma_i^2 + \mu_i^2 - \log(\sigma_i^2) - 1 \right)
$$

## 重参数化技巧
VAE的一个重要创新是重参数化技巧（Reparameterization Trick）。由于KL散度涉及到潜在变量的分布，它在反向传播时不易计算。为了解决这个问题，VAE将潜在变量的采样过程变得可微分，使得梯度可以有效地传播。
具体地，假设编码器输出的是潜在变量的均值$μ$和标准差 $σ$，VAE通过以下重参数化技巧将潜在变量 $z$ 表达为：
$$z=μ+σ⋅ϵ$$

其中 $ϵ∼N(0,I)$ 是从标准正态分布中采样的噪声。这样，尽管潜在变量的分布是通过 $μ$ 和 $σ$ 参数化的，采样过程仍然是可微的，从而可以通过反向传播进行训练。
![[Pasted image 20260411221528.png]]


![[屏幕截图 2026-04-11 210309.png]]

# VAE代码示例！！！！
```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np

# 设置超参数
batch_size = 128
epochs = 10
learning_rate = 1e-3
latent_dim = 20  # 潜在空间的维度

# 数据预处理（使用MNIST数据集）
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# 定义编码器（Encoder）
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # 潜在变量的均值
        self.fc22 = nn.Linear(400, latent_dim)  # 潜在变量的标准差

    def forward(self, x):
        h1 = torch.relu(self.fc1(x.view(-1, 28*28)))
        z_mean = self.fc21(h1)
        z_log_var = self.fc22(h1)
        return z_mean, z_log_var

# 定义解码器（Decoder）
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28*28)

    def forward(self, z):
        h3 = torch.relu(self.fc3(z))
        reconstruction = torch.sigmoid(self.fc4(h3))
        return reconstruction

# 定义VAE（包括编码器、解码器及重参数化）
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var

# 定义损失函数
def loss_function(reconstruction, x, z_mean, z_log_var):
    BCE = nn.functional.binary_cross_entropy(reconstruction, x.view(-1, 28*28), reduction='sum')
    # KL散度
    # p(z) ~ N(0, I), q(z|x) ~ N(mu, sigma^2)
    # D_KL(q(z|x) || p(z)) = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    # 其中，mu和sigma是从编码器获得的
    # z_mean是mu，z_log_var是log(sigma^2)
    # 因此，KL散度是：
    # KL = -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KL = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
    return BCE + KL

# 初始化模型和优化器
model = VAE(latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 训练过程
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        reconstruction, z_mean, z_log_var = model(data)
        loss = loss_function(reconstruction, data, z_mean, z_log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.6f}")
    
    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")

# 生成图像
def generate_images(epoch, num_images=10):
    model.eval()
    with torch.no_grad():
        # 随机生成潜在变量
        z = torch.randn(num_images, latent_dim).to(device)
        sample = model.decoder(z).cpu()
        sample = sample.view(num_images, 28, 28)
        
        # 显示生成的图像
        fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
        for i in range(num_images):
            axes[i].imshow(sample[i], cmap='gray')
            axes[i].axis('off')
        plt.savefig(f"generated_images_epoch_{epoch}.png")
        plt.close()

# 定义设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# 训练VAE
for epoch in range(1, epochs + 1):
    train(epoch)
    generate_images(epoch)

```

#### 代码解释！！！
下面是该 VAE 代码的逐行解释，涵盖导入、参数设置、数据加载、网络模块定义、损失函数、训练循环及图像生成。

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
```
- 导入必要的库：`torch` 核心库，`nn` 用于构建神经网络，`optim` 提供优化器，`DataLoader` 用于批量加载数据，`datasets` 和 `transforms` 用于处理 MNIST 数据集，`matplotlib.pyplot` 用于显示/保存图像，`numpy` 用于数值操作（本代码中未显式使用，但常作为辅助）。

```python
batch_size = 128
epochs = 10
learning_rate = 1e-3
latent_dim = 20
```
- 设置超参数：批大小 128，训练轮数 10，学习率 0.001，潜在空间维度 20。

```python
transform = transforms.ToTensor()
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
```
- `ToTensor()`：将 PIL 图像或 numpy 数组转换为形状为 `(C, H, W)` 的 torch 张量，并将像素值从 `[0,255]` 缩放到 `[0,1]`。
- 加载 MNIST 训练集（`train=True`），下载到 `./data` 目录，应用上述变换。
- 创建 DataLoader，按 `batch_size` 打乱数据并提供迭代接口。

```python
class Encoder(nn.Module):
    def __init__(self, latent_dim):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(28*28, 400)
        self.fc21 = nn.Linear(400, latent_dim)  # 均值
        self.fc22 = nn.Linear(400, latent_dim)  # 对数方差
```
- 定义编码器类，继承自 `nn.Module`。
- 构造方法：调用父类初始化；创建三个全连接层：
  - `fc1`：输入 784 维（28×28 展平），输出 400 维，后接 ReLU。
  - `fc21`：输出潜在分布的均值 `mu`，维度 `latent_dim`。
  - `fc22`：输出潜在分布的对数方差 `log_var`，维度 `latent_dim`。

```python
    def forward(self, x):
        h1 = torch.relu(self.fc1(x.view(-1, 28*28)))
        z_mean = self.fc21(h1)
        z_log_var = self.fc22(h1)
        return z_mean, z_log_var
```
- 前向传播：
  - `x.view(-1, 28*28)`：将输入图像展平，第一维保持批大小。
  - 通过 `fc1` 后接 ReLU 激活得到 `h1`。
  - 分别通过 `fc21` 和 `fc22` 得到均值和对数方差。
- 返回 `(z_mean, z_log_var)`。

```python
class Decoder(nn.Module):
    def __init__(self, latent_dim):
        super(Decoder, self).__init__()
        self.fc3 = nn.Linear(latent_dim, 400)
        self.fc4 = nn.Linear(400, 28*28)
```
- 解码器类：构造方法定义两个全连接层。
  - `fc3`：将潜在向量（`latent_dim`）映射到 400 维。
  - `fc4`：将 400 维映射回 784 维（重构的图像像素）。

```python
    def forward(self, z):
        h3 = torch.relu(self.fc3(z))
        reconstruction = torch.sigmoid(self.fc4(h3))
        return reconstruction
```
- 前向传播：`z` 通过 `fc3` + ReLU，再通过 `fc4` + sigmoid 激活，输出范围在 (0,1) 之间，适合二值图像（MNIST 为灰度，但使用 sigmoid 模拟像素概率）。

```python
class VAE(nn.Module):
    def __init__(self, latent_dim):
        super(VAE, self).__init__()
        self.encoder = Encoder(latent_dim)
        self.decoder = Decoder(latent_dim)
```
- VAE 主类：组合编码器和解码器。

```python
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
```
- 重参数化技巧：从 `N(mu, exp(logvar))` 中采样，但将随机性转移到标准正态噪声 `eps` 上，使梯度能通过 `mu` 和 `logvar` 回传。
  - `std = exp(0.5 * logvar)` 计算标准差。
  - `eps ~ N(0,1)`，形状与 `std` 相同。
  - 返回 `mu + eps * std`。

```python
    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize(z_mean, z_log_var)
        reconstruction = self.decoder(z)
        return reconstruction, z_mean, z_log_var
```
- VAE 前向传播：
  - 编码器输出均值和 log 方差。
  - 重参数化得到潜在向量 `z`。
  - 解码器重构图像`reconstruction`。
- 返回重构图像、均值、对数方差（用于计算 KL 散度）。

```python
def loss_function(reconstruction, x, z_mean, z_log_var):
    BCE = nn.functional.binary_cross_entropy(reconstruction, x.view(-1, 28*28), reduction='sum')
```
- 定义 VAE 损失函数：重构误差 + KL 散度。
  - `binary_cross_entropy`：计算重构与原始输入之间的二值交叉熵。`reduction='sum'` 表示对 batch 内所有像素求和（而不是平均）。

```python
    KL = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - torch.exp(z_log_var))
    return BCE + KL
```
- KL 散度：`KL(q(z|x) || p(z))`，其中 `p(z)=N(0,I)`，`q(z|x)=N(z_mean, exp(z_log_var))`。
  - 解析公式：`-0.5 * Σ(1 + log(σ²) - μ² - σ²)`，这里 `z_log_var` 就是 `log(σ²)`，`torch.exp(z_log_var)` 是 σ²。
  - 对 batch 内所有样本所有维度求和。
- 返回总损失（标量）。

```python
model = VAE(latent_dim)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
```
- 实例化 VAE 模型，创建 Adam 优化器。

```python
def train(epoch):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
```
- 训练函数：`model.train()` 设置为训练模式（影响 dropout 等，本代码无影响）。
- 遍历 DataLoader，`data` 为图像张量，`_` 为标签（未使用）。
- 将数据移动到指定设备（GPU/CPU）。

```python
        optimizer.zero_grad()
        reconstruction, z_mean, z_log_var = model(data)
        loss = loss_function(reconstruction, data, z_mean, z_log_var)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
```
- 梯度清零 → 前向传播 → 计算损失 → 反向传播 → 累加损失（标量值）→ 更新参数。

```python
        if batch_idx % 100 == 0:
            print(f"Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)}] Loss: {loss.item() / len(data):.6f}")
```
- 每 100 个 batch 打印一次进度：当前 epoch、已处理的样本数、该 batch 的平均损失（总损失除以 batch_size）。

```python
    print(f"====> Epoch: {epoch} Average loss: {train_loss / len(train_loader.dataset):.4f}")
```
- 每个 epoch 结束后打印平均损失（总损失之和 / 数据集总样本数）。

```python
def generate_images(epoch, num_images=10):
    model.eval()
    with torch.no_grad():
        z = torch.randn(num_images, latent_dim).to(device)
        sample = model.decoder(z).cpu()
        sample = sample.view(num_images, 28, 28)
```
- 图像生成函数：`model.eval()` 设为评估模式（禁用 dropout 等）。
- `no_grad()` 禁用梯度计算，节省内存。
- 从标准正态分布采样 `num_images` 个潜在向量，送到解码器生成图像，移回 CPU，并 reshape 为 `(num_images, 28, 28)`。

```python
        fig, axes = plt.subplots(1, num_images, figsize=(15, 15))
        for i in range(num_images):
            axes[i].imshow(sample[i], cmap='gray')
            axes[i].axis('off')
        plt.savefig(f"generated_images_epoch_{epoch}.png")
        plt.close()
```
- 创建 1 行 `num_images` 列的子图，每张图显示一张灰度图像，关闭坐标轴。
- 保存为 PNG 文件，关闭图形以释放内存。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
```
- 自动选择可用设备（GPU 优先），将模型参数移动到该设备。

```python
for epoch in range(1, epochs + 1):
    train(epoch)
    generate_images(epoch)
```
- 主循环：依次训练每个 epoch，并在每个 epoch 结束后生成一组图像（保存为文件）。

**注意事项**：
- 代码中 `train` 函数内部使用了变量 `device`，但 `device` 的定义在函数之后。由于 Python 闭包在运行时才查找 `device`，且调用 `train` 时 `device` 已经定义，因此可以正常工作。更规范的做法是将 `device` 作为参数传入或定义在全局最顶部。
- `loss_function` 中 `BCE` 的输入 `x.view(-1, 28*28)` 应与 `reconstruction` 形状一致，均为 `(batch_size, 784)`，正确。