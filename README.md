# FGIC-ViT-ML

Exploring Vision Transformer Embeddings with Classical Machine Learning for Fine-Grained Image Recognition.

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Table of Contents
- [About The Project](#about-the-project)
  - [Built With](#built-with)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Data Setup](#data-setup)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Citation](#citation)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)

## About The Project

FGIC-ViT-ML is a comprehensive framework for fine-grained image classification that combines the power of Vision Transformers (ViT) for feature extraction with traditional machine learning algorithms. The project focuses on processing and analyzing six different fine-grained image datasets using various state-of-the-art ViT models.

### Built With

* [Python](https://www.python.org/)
* [PyTorch](https://pytorch.org/)
* [CLIP](https://github.com/openai/CLIP)
* [DINOv2](https://github.com/facebookresearch/dino)
* [PyCaret](https://pycaret.org/)
* [Pandas](https://pandas.pydata.org/)
* [scikit-learn](https://scikit-learn.org/)

## Getting Started

Follow these steps to set up the project locally.

### Prerequisites

* Python 3.8 or higher
* CUDA-capable GPU (recommended)
* At least 100GB of free disk space

### Installation

1. Clone the repository
```bash
git clone https://github.com/dmjimenezbravo/FGIC-ViT-ML.git
cd FGIC-ViT-ML
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

### Data Setup

1. Download and set up datasets following instructions in `data/raw/README.md`
2. Process the datasets following `data/processed/README.md`
3. Generate embeddings following `data/embeddings/README.md`

## Project Structure

```
FGIC-ViT-ML/
├── data/                   # Data directory
│   ├── raw/               # Original datasets
│   ├── processed/         # Processed datasets
│   └── embeddings/        # Generated embeddings
├── notebooks/             # Jupyter notebooks
│   ├── 1_datasets_exploration_visualization.ipynb
│   ├── 2_embeddings_extraction.ipynb
│   └── 3_train_evaluation.ipynb
├── src/                   # Source code
│   ├── data/             # Dataset processing
│   ├── ml_classifiers/   # ML training and evaluation
│   └── vit_embeddings/   # ViT embedding extraction
└── results/              # Training results and evaluations
```

## Usage

1. First, explore the datasets:
```bash
jupyter notebook notebooks/1_datasets_exploration_visualization.ipynb
```

2. Extract embeddings from ViT models:
```bash
jupyter notebook notebooks/2_embeddings_extraction.ipynb
```

3. Train and evaluate ML models:
```bash
jupyter notebook notebooks/3_train_evaluation.ipynb
```

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any contributions you make are greatly appreciated.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

Distributed under the GNU General Public License v3.0. See `LICENSE` for more information.

## Citation

If you use this code in your research, please cite our paper:

```bibtex
@article{jimenezbravo2025fgic,
  title={Beyond End-to-End Learning: Exploring Vision Transformer Embeddings with Classical Machine Learning for Fine-Grained Image Classification},
  author={Jim{\'e}nez-Bravo, Diego M.; Haro Crespo, V{\'i}ctor; Navarro-C{\'a}ceres, Juan Jos{\'e}; S{\'a}nchez San Blas, H{\'e}ctor; Sales Mendes, Andr{\'e} Filipe},
  journal={[Journal Pending]},
  year={2025},
  publisher={[Publisher Pending]},
  doi={[DOI Pending]}
}
```

## Contact

Diego M. Jiménez Bravo - [@dmjimenezbravo](https://twitter.com/dmjimenezbravo) - dmjimenezbravo@gmail.com

Project Link: [https://github.com/dmjimenezbravo/FGIC-ViT-ML](https://github.com/dmjimenezbravo/FGIC-ViT-ML)

## Acknowledgments

* Vision Transformer Models:
  * [CLIP](https://github.com/openai/CLIP)
  * [DINOv2](https://github.com/facebookresearch/dino)
  * [Franca](https://huggingface.co/docs/transformers/model_doc/franca)
  * [SigLIPv2](https://huggingface.co/docs/transformers/model_doc/siglip)
* Dataset Providers:
  * Caltech-UCSD Birds-200-2011
  * FGVC Aircraft
  * NABirds
  * Oxford 102 Flowers
  * Stanford Cars
  * Stanford Dogs
