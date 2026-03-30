import numpy as np

def pca(data: np.ndarray, k: int) -> np.ndarray:
    """
    Perform PCA and return the top k principal components.
    
    Args:
        data: Input array of shape (n_samples, n_features)
        k: Number of principal components to return
    
    Returns:
        Principal components of shape (n_features, k), rounded to 4 decimals.
        Each eigenvector's sign is fixed so its first non-zero element is positive.
    """
    # Your code here
    mean=np.mean(data,axis=0)
    #统一量纲标准化
    std=np.std(data,axis=0)
    std_adj=np.where(std<1e-10,1.0,std)
    data_norm=(data-mean)/std_adj #标准化
    #计算协方差矩阵
    cov_matrix=np.cov(data_norm,rowvar=False)#行为样本，列为特征
    #计算特征向量，特征值。特征向量是竖着排列的
    eigenvalues,eigenvectors=np.linalg.eigh(cov_matrix)
    #降序排序特征向量和特征值,取方差最大的向量方向作为主成分
    idx=np.argsort(eigenvalues)[::-1]
    eigenvalues=eigenvalues[idx]
    eigenvectors=eigenvectors[:,idx]
    #取前k个主成分
    top_k_eigenvectors=eigenvectors[:, :k]
    #符号修正
    for j in range(k):
        v=top_k_eigenvectors[:,j]
        for i in range(len(v)):
            if np.abs(v[i])>1e-10:
                if v[i]<0:
                    top_k_eigenvectors[:,j] *= -1
                break
    return np.round(top_k_eigenvectors,4) #保留小数点后4位
