import tqdm
import numpy as np

__all__ = ['transrate']

def scale_trace(Z):
    factor = 1 / np.sqrt(np.trace(Z @ Z.transpose()))
    return Z * factor

def coding_rate(Z, eps = 1e-4):
    n, d = Z.shape
    (_, rate) = np.linalg.slogdet((np.eye(d) + 1 / (n * eps) * Z.transpose() @ Z))
    return 0.5 * rate
    
def transrate(Z, y, eps = 1e-4):    
    Z = scale_trace(Z)
    Z = Z - np.mean(Z, axis=0, keepdims=True)    
    RZ = coding_rate(Z, eps)
    RZY = 0.
    K = int(y.max() + 1)
    for i in range(K):
        RZY += coding_rate(Z[(y==i).flatten()], eps)
    return RZ - RZY/K


if __name__=='__main__':
    features = np.random.randn(100000,1024)
    labels = np.array(np.arange(100).tolist() * 1000)
    print(features.shape, labels.shape)
    for eps in [1e0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9 ]:
        print(eps, transrate(features, labels, eps))
