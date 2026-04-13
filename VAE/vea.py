import numpy as np

def vae_loss(x: np.ndarray, x_reconstructed: np.ndarray, mu: np.ndarray, log_var: np.ndarray) -> tuple:
    recon_loss=np.mean(np.sum((x-x_reconstructed)**2, axis=1))
    KL=-0.5*np.mean(np.sum((1+log_var-np.square(mu)-np.exp(log_var)), axis=1))
    return float(recon_loss+KL),float(recon_loss), float(KL)
