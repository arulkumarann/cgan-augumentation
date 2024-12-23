import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from ..cgan.models.generator import Generator

def visualize_synthetic_data(generator, labels_one_hot, z_dim=100, num_samples=1000):
    z = torch.randn(num_samples, z_dim)
    synthetic_images = generator(z, labels_one_hot).detach().numpy()
    synthetic_images_flattened = synthetic_images.reshape(synthetic_images.shape[0], -1)
    
    tsne = TSNE(n_components=2)
    reduced_data = tsne.fit_transform(synthetic_images_flattened)

    plt.scatter(reduced_data[:, 0], reduced_data[:, 1])
    plt.title("t-SNE Visualization of Synthetic Images")
    plt.show()

if __name__ == "__main__":
    generator = Generator(z_dim=100, num_classes=10)
    labels_one_hot = torch.eye(10)  # Example for one-hot encoding
    visualize_synthetic_data(generator, labels_one_hot)
