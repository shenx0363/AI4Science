# Lightning重新封装方案
---
# model.py
包含自编码器模型：AutoEncoder    
提供：前向传播forward_mask()，损失函数loss_mask()，特征提取feature()

### 封装方案：  
写一个**训练控制器**：前向传播、训练、验证、优化  
封装一个模型类`ScMAELightning`，父类：`LightningModule`

包含以下Lightning生命周期函数：
1. `__init__()` ：构造  
		保存超参数(`save_hyperparameters`)->定义model->定义掩码概率
2. `forward()` ：前向传播  
		返回降维后的潜在变量特征Latent features
3. `training_step()` :训练  
		取数据(batch中的x)->加噪->算损失->日志
4. `validation_step()` ：验证  
		取Latent特征和对应标签y，用来评估
5. `configure_optimizers(self)` :优化器  
		从`self.hparams` 中读取 `learning_rate` ，Adam优化更新model所有参数
6. `on_validation_epoch_end` :汇总验证  
		80轮做一次整体评估

# dateset.py
分装一个加载数据集的类`ScMAEDataModule`，父类：`LightningDataModule`

参数中心化->数据集预处理->加载数据给模型(batch)  
`__init__()` -> `setup(stage)` -> `train_dataloader` 

# main.py
调数据集->调模型->回调，自动监控，自动保存->GPU训练  
自动处理 `.to(device)`，支持多卡 (DDP) 或 TPU 无缝切换


LightningModule->LightningDataModule->Trainer  
`trainer.fit()` 替代了原有的双重嵌套 `for` 循环。  
Trainer**内置生命周期**，它自动管理了 `train`、`val` 和 `test` 的切换。


最后，在requirements.txt文件中添加
```
pytorch-lightning==2.6.1
```
