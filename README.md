# Single Image Haze Removal using a GAN

This repository contains an implementation of the paper ["Single Image Haze Removal using a Generative Adversarial Network"](https://arxiv.org/abs/1810.09479) for image dehazing.
Output test folder: https://drive.google.com/file/d/1qFU92-fII6bDWWOJHMyS89xtJUGQtoro/view?usp=sharing 

## Model Architecture

The model uses a GAN-based approach with the following components:

- **Generator**: A Tiramisu (FC-DenseNet) architecture with 56 layers, which effectively learns the mapping from hazy to clear images. The network features dense blocks with skip connections between encoder and decoder paths to preserve fine details.

- **Discriminator**: A PatchGAN discriminator that classifies if overlapping image patches are real or fake, enabling the model to capture high-frequency structure.

- **VGG19 Feature Extractor**: Used for perceptual loss calculation to improve visual quality and structural similarity.

## Loss Functions

The model is trained with multiple loss components:

- **GAN Loss**: Adversarial loss to make generated images indistinguishable from real ones
- **L1 Loss**: Pixel-wise reconstruction loss to ensure content preservation
- **VGG Perceptual Loss**: Feature-space similarity to enhance perceptual quality

## Training Details (dataset link=>"https://www.kaggle.com/datasets/kmljts/reside-6k")

- **Dataset**: RESIDE-6K (6,000 synthetic hazy images paired with clear ground truth)
- **Hardware**: CUDA-enabled GPU
- **Epochs**: 6 (can be extended for better results)
- **Batch Size**: 1
- **Learning Rate**: 0.001
- **Optimizer**: Adam (β1=0.5, β2=0.999)

## Results

After training for 6 epochs, the model achieved:

- **PSNR**: 15.21 dB
- **SSIM**: 0.59
- **Combined Score**: 1.35
  (later i will modify the code and upload the code with better results)
### Training Progress

The training graphs show:

1. **Generator Losses**: 
   - Total generator loss stabilizes around 39
   - L1 loss decreases significantly in early epochs and then stabilizes (~31)
   - GAN loss increases gradually as the discriminator becomes more challenging

2. **Discriminator Loss**: 
   - Decreases steadily from 0.51 to 0.19, indicating improved discrimination ability

3. **Validation Metrics**:
   - PSNR decreases slightly from 15.45 to 15.21
   - SSIM improves significantly from 0.41 to 0.59
   - Overall score increases from 1.19 to 1.35

## Usage

The implementation includes classes for:
- Custom dataset loading and processing
- Model definition and training
- Evaluation and inference
- Visualization utilities

## Requirements

- PyTorch
- torchvision
- NumPy
- Matplotlib
- scikit-image
- tqdm

## Citation

```
@article{liu2019single,
  title={Single Image Haze Removal using a Generative Adversarial Network},
  author={Liu, Yingbiao and Wang, Xin and Zhu, Lei and Liu, Yunhong},
  journal={arXiv preprint arXiv:1810.09479},
  year={2018}
}
```
