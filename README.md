# GAN-based Augmentation for Imbalanced Datasets

This project uses a **Conditional GAN (cGAN)** to generate synthetic images for underrepresented classes in an imbalanced dataset. The goal is to balance the dataset and improve classification performance.

## What is cGAN?

A **Conditional GAN (cGAN)** is a type of GAN where the generator creates images based on specific conditions (e.g., class labels). The generator learns to create realistic images for each class, while the discriminator distinguishes between real and generated images. This helps in augmenting the dataset, especially for underrepresented classes.

## Files

- **`train.py`**: Trains the cGAN model, generating synthetic images for minority classes.
- **`models.py`**: Defines the architecture of the generator and discriminator networks.
- **`utils.py`**: Contains helper functions for data handling, saving generated images, and visualizations.
- **`evaluate.py`**: Evaluates model performance using augmented data and compares classifier results.
## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/arulkumarann/cgan-augumentation.git
   cd gan-augmentation
2. Install dependencies:
    ```bash 
    pip install -r requirements.txt

## Usage

### Prepare Data
Use any imbalanced dataset (CIFAR-10 by default).

### Train Model
    python train.py


## License
This project is licensed under the MIT License.








