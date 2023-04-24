from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from matplotlib import pyplot as plt

def t_SNE_image(data_x, y):
    tsne = TSNE(random_state = 42, n_components=2,verbose=0, perplexity=40, n_iter=300)
    tsne_test= tsne.fit_transform(data_x)
    figure = plt.figure(figsize=(7,5))
    plt.scatter(tsne_test[:, 0], tsne_test[:, 1], s= 5, c=y, cmap='Spectral')
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(11)-0.5).set_ticks(np.arange(10))
    plt.title('Visualizing MNIST through t-SNE', fontsize=16)
    return figure