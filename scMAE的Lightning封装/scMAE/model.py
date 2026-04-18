import torch
import torch.nn as nn
# 二分类交叉熵-mask预测
from torch.nn.functional import binary_cross_entropy_with_logits as bce_logits
# 均方误差-重构损失
from torch.nn.functional import mse_loss as mse
import pytorch_lightning as pl
from datasets import apply_noise
from torch.optim import Adam
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from evaluate import evaluate
class AutoEncoder(torch.nn.Module):
    def __init__(
        self,
        num_genes,
        hidden_size=128,
        dropout=0,
        masked_data_weight=.75, # 被mask掉的数据在重构损失中的权重，默认为0.75，表示被mask掉的数据在重构损失中占75%的权重，未被mask掉的数据占25%的权重
        mask_loss_weight=0.7, # mask预测损失在总损失中的权重，默认为0.7，表示mask预测损失占总损失的70%，重构损失占30%
    ):
        super().__init__()
        self.num_genes = num_genes
        self.masked_data_weight = masked_data_weight
        self.mask_loss_weight = mask_loss_weight

        self.encoder = nn.Sequential(
            nn.Dropout(p=dropout), # dropout层，随机将输入的一部分元素设置为0，以防止过拟合
            nn.Linear(self.num_genes, 256),
            nn.LayerNorm(256),
            nn.Mish(inplace=True), # Mish激活函数，Mish是一种平滑的非线性激活函数，定义为x * tanh(softplus(x))，其中softplus(x) = log(1 + exp(x))，Mish在某些任务中表现出比ReLU更好的性能
            nn.Linear(256, hidden_size),
            nn.LayerNorm(hidden_size), # 256 → latent维度
            nn.Mish(inplace=True),
            nn.Linear(hidden_size, hidden_size)
        )

        self.mask_predictor = nn.Linear(hidden_size, num_genes)
        self.decoder = nn.Linear(
            in_features=hidden_size+num_genes, out_features=num_genes)

    def forward_mask(self, x):
        latent = self.encoder(x)
        predicted_mask = self.mask_predictor(latent)
        reconstruction = self.decoder(
            torch.cat([latent, predicted_mask], dim=1))

        return latent, predicted_mask, reconstruction

    def loss_mask(self, x, y, mask):
        latent, predicted_mask, reconstruction = self.forward_mask(x)
        w_nums = mask * self.masked_data_weight + (1 - mask) * (1 - self.masked_data_weight)
        reconstruction_loss = (1-self.mask_loss_weight) * torch.mul(
            w_nums, mse(reconstruction, y, reduction='none'))

        mask_loss = self.mask_loss_weight * \
            bce_logits(predicted_mask, mask, reduction="mean")
        reconstruction_loss = reconstruction_loss.mean()

        loss = reconstruction_loss + mask_loss 
        return latent, loss

    def feature(self, x):
        latent = self.encoder(x)
        return latent

# Lightning封装模型类
class ScMAELightning(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args) # 保存超参数，方便后续加载模型时使用相同的超参数
        self.model = AutoEncoder(
            num_genes=args["data_dim"],
            hidden_size=128,
            masked_data_weight=0.75,
            mask_loss_weight=0.7,
        )
        self.mask_probas = [0.4] * args["data_dim"] # 每个基因被mask掉的概率，默认为0.4，表示每个基因有40%的概率被mask掉
    
    def forward(self, x):
        return self.model.feature(x) # 前向传播，返回编码器输出的latent特征
    
    def training_step(self, batch, batch_idx):
        x, _ =batch
        x_corrupted, mask = apply_noise(x, self.mask_probas) # 对输入数据添加噪声，生成被mask掉的数据和对应的mask
        _, loss = self.model.loss_mask(x_corrupted, x, mask) # 计算损失，输入被mask掉的数据、原始数据和mask
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True) # 日志，记录训练损失，方便后续分析和可视化
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        latent = self.model.feature(x)
        return {"latent": latent, "label": y} # 返回编码器输出的latent特征和对应的标签，方便后续评估模型性能
    
    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.hparams["learning_rate"])

    def on_validation_epoch_end(self):
        # 1. 检查是否达到了评估 Epoch，且列表不为空
        if self.current_epoch == 80 and len(self.validation_step_outputs) > 0:
            
            # 2. 提取并拼接所有 batch 的数据
            latents = torch.cat([o["latent"] for o in self.validation_step_outputs], dim=0).cpu().numpy()
            labels = torch.cat([o["label"] for o in self.validation_step_outputs], dim=0).cpu().numpy()
            
            # 3. 聚类评估逻辑
            from evaluate import evaluate # 确保 evaluate.py 在路径中
            
            if latents.shape[0] < 10000:
                clustering_model = KMeans(n_clusters=self.hparams["n_classes"], n_init='auto')
                clustering_model.fit(latents)
                pred_label = clustering_model.labels_
            else:
                import scanpy as sc
                adata = sc.AnnData(latents)
                sc.pp.neighbors(adata, n_neighbors=10, use_rep="X")
                # 假设 res_search_fixed_clus 已在外部定义或导入
                reso = res_search_fixed_clus(adata, self.hparams["n_classes"])  
                sc.tl.leiden(adata, resolution=reso)
                pred_label = [int(x) for x in adata.obs['leiden'].to_list()]
            
            # 4. 计算指标
            nmi, ari, acc = evaluate(labels, pred_label)
            sil = silhouette_score(latents, pred_label)
            
            # 5. 日志记录
            self.log_dict({
                "val/nmi": nmi,
                "val/ari": ari,
                "val/acc": acc,
                "val/sil": sil
            }, sync_dist=True) # sync_dist=True 适配多卡训练

        # --- 重要：每一轮验证结束都要清空列表，释放内存 ---
        self.validation_step_outputs.clear()
            