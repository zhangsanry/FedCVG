# FedCVG: A Two-Stage Robust Federated Learning Optimization Algorithm

FedCVG is a federated learning algorithm, a two-stage robust framework against poisoning attacks. The first stage involves identifying malicious nodes based on reputation mechanisms and K-means clustering, aiming to detect and remove malicious clients conducting poisoning attacks in data heterogeneity environments. The second stage decouples the feature extraction and classifier training processes, and achieves "virtual aggregation" of clients by recording historical gradients. This project implements FedCVG and provides a comparative experimental framework with various existing federated learning algorithms (such as FedAvg, SCAFFOLD, FedProx, etc.).

## Project Directory Structure

```
.
├── consumption/        # Communication and computational resource consumption monitoring module
├── data/               # Dataset storage directory (automatically downloaded at runtime)
├── figures/            # Experimental results charts and visualization output directory
├── models/             # Neural network model definitions (CNN, LeNet, ResNet, etc.)
├── preprocessing/      # Data preprocessing and dataset partitioning module
├── results/            # Experimental results storage directory
├── visualization/      # Result visualization tools and scripts
├── fedcvg.py           # FedCVG algorithm core implementation
├── federated_learning.py   # Basic federated learning framework implementation
├── federated_parser.py     # Command line argument parser
├── main.py             # Standard federated learning entry point
├── main_fedcvg.py      # FedCVG algorithm dedicated entry point
├── requirements.txt    # Project dependencies list
└── LICENSE             # Open source license
```

## Main Module Description

### Core Modules

- **fedcvg.py**: Implements the core logic of the FedCVG algorithm, including client verification mechanisms and gradient memory functionality.
- **federated_learning.py**: Implements the basic federated learning framework, including client selection, model aggregation, and other general features.
- **federated_parser.py**: Parses command line arguments, configures experimental parameters such as dataset selection, model type, number of clients, etc.
- **main.py**: Program entry point for standard federated learning algorithms.
- **main_fedcvg.py**: Dedicated entry point for the FedCVG algorithm, configured with specific parameters and features.

### Model Module (models/)

Contains definitions of various deep learning models for federated learning experiments:
- CNN: Basic CNN network for datasets like CIFAR10
- LeNet: Classic LeNet-5 network, suitable for datasets like MNIST
- ResNet: Includes variants such as ResNet18, ResNet34, for more complex image classification tasks

### Data Preprocessing (preprocessing/)

- **baselines_dataloader.py**: Implements dataset loading and data partitioning functionality, supporting both IID and Non-IID data distributions
  - IID: Independent and identically distributed partitioning, where each client receives randomly and uniformly distributed data
  - Non-IID: Non-independent and identically distributed partitioning, using Dirichlet distribution to control label skew

### Communication and Computation Consumption Monitoring (consumption/)

Monitors and records communication and computational resource consumption during the federated learning process:
- **fedavg_comm.py**: Communication consumption tracking for the FedAvg algorithm
- **auror_comm.py**: Communication consumption tracking for the Auror algorithm
- **fedcvg_comm.py**: Communication consumption tracking for the FedCVG algorithm

### Visualization Tools (visualization/)

A toolkit for result analysis and visualization:
- **visualize_model_results.py**: Generates model accuracy curve charts
- **table_data_collector.py**: Extracts tabular data from experimental results
- **plot_3d_bar.py**: Creates 3D bar charts comparing the performance of different algorithms under various data distributions
- **comm_vs_accuracy.py**: Plots the relationship between communication overhead and accuracy
- **run_plots.py**: Batch running tool for visualization scripts

## How to Use

### Environment Setup

```bash
pip install -r requirements.txt
```

### Running Standard Federated Learning Experiments

```bash
python main.py --dataset MNIST --model LeNet --num_client 10 --num_local_class 2 --num_round 100
```

### Running FedCVG Experiments

```bash
python main_fedcvg.py --dataset CIFAR10 --model ResNet34 --num_client 20 --alpha 0.5 --num_round 100
```

### Parameter Description

- `--dataset`: Select dataset (MNIST, FashionMNIST, CIFAR10, etc.)
- `--model`: Select model (LeNet, ResNet34, etc.)
- `--num_client`: Number of clients participating in federated learning
- `--num_round`: Number of federated learning rounds
- `--alpha`: Parameter controlling the degree of Non-IID (smaller values indicate more imbalanced data distribution)
- `--algorithm`: Select federated learning algorithm (FedAvg, FedProx, SCAFFOLD, FedCVG, etc.)

### Visualizing Results

```bash
python visualization/run_visualization.py --results_dir ./results --output_dir ./figures
```

## Supported Algorithms

- **FedAvg**: Basic federated averaging algorithm
- **FedProx**: Federated learning with proximal regularization
- **SCAFFOLD**: Federated learning with control variables
- **FedCVG**: Client verification and gradient memory federated learning proposed in this project
- **Krum/MultiKrum**: Robust aggregation algorithms against Byzantine attacks
- **Median/TrimmedMean**: Statistically-based robust aggregation algorithms
- **Auror**: Clustering-based anomaly detection algorithm

## Citation

If you use the FedCVG algorithm or this codebase in your research, please consider citing our work.

## License

This project is under an open source license, please see the LICENSE file for details. 