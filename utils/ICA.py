from sklearn.decomposition import FastICA
from matplotlib import pyplot as plt
import numpy as np

def ICA(x, n_components=2):
    ica = FastICA(n_components=n_components)
    sources = ica.fit_transform(x)
    return sources

if __name__ == '__main__':
    np.random.seed(0)
    S = np.random.normal(size=(200, 2))
    print(S.T.shape)
    A = np.array([[1, 1], [0, 2]])  # mixing matrix
    X = np.dot(S, A.T)

    # Plots of signals before mixing
    plt.figure(1)
    plt.subplot(6, 1, 1)
    plt.plot(S.T[0])
    plt.subplot(6, 1, 2)
    plt.plot(S.T[1])

    # Plots of signals after mixing
    plt.subplot(6, 1, 3)
    plt.plot(X.T[0], color='red')
    plt.subplot(6, 1, 4)
    plt.plot(X.T[1], color='red')
    
    # Plots of Indipendent Components
    sources = ICA(X, 2)
    plt.subplot(6, 1, 5)
    plt.plot(sources.T[0])
    plt.subplot(6, 1, 6)
    plt.plot(sources.T[1])
    plt.show()

    print(np.allclose(S, sources))
