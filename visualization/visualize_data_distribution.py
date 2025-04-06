import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import argparse
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, FashionMNIST, MNIST
from torch.utils.data import DataLoader, Subset, Dataset
import random
import json
import seaborn as sns

def get_dataset(dataset_name, root_dir='./data'):
    """Load the specified dataset"""
    if dataset_name.upper() == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_dataset = MNIST(root=root_dir, train=True, download=True, transform=transform)
        classes = list(range(10))
        
    elif dataset_name.upper() == 'FASHIONMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.2860,), (0.3530,))
        ])
        train_dataset = FashionMNIST(root=root_dir, train=True, download=True, transform=transform)
        classes = list(range(10))
        
    elif dataset_name.upper() == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
        ])
        train_dataset = CIFAR10(root=root_dir, train=True, download=True, transform=transform)
        classes = list(range(10))
        
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    return train_dataset, classes

def distribute_data_noniid_label(dataset, num_clients, alpha, seed=42):
    """
    Distribute data to clients in a non-IID fashion using Dirichlet distribution
    
    Parameters:
    - dataset: Dataset to distribute
    - num_clients: Number of clients
    - alpha: Parameter controlling non-IID degree (smaller means more skewed)
    - seed: Random seed
    
    Returns:
    - client_idxs: Dictionary containing data indices for each client
    - distribution_stats: Statistics about data distribution
    """
    np.random.seed(seed)
    random.seed(seed)
    
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))
    
    # Initialize client data indices dictionary
    client_idxs = {i: [] for i in range(num_clients)}
    
    # Generate Dirichlet distribution for each class
    label_distribution = np.random.dirichlet([alpha] * num_clients, num_classes)
    
    # Organize data indices by class
    class_idxs = {k: np.where(labels == k)[0] for k in range(num_classes)}
    
    # Allocate data to clients
    for k in range(num_classes):
        class_size = len(class_idxs[k])
        # Get proportion of class k data for each client
        proportions = label_distribution[k]
        # Calculate actual allocation numbers
        proportions = np.array([p * class_size for p in proportions])
        proportions = proportions.astype(int)
        # Handle rounding difference
        diff = class_size - sum(proportions)
        # Distribute difference to random clients
        proportions[np.random.choice(num_clients, diff, replace=False)] += 1
        
        # Allocate indices
        class_idxs_shuffled = class_idxs[k].copy()
        np.random.shuffle(class_idxs_shuffled)
        
        # Assign indices to clients
        start_idx = 0
        for i in range(num_clients):
            end_idx = start_idx + proportions[i]
            client_idxs[i].extend(class_idxs_shuffled[start_idx:end_idx])
            start_idx = end_idx
    
    # Calculate statistics for each client's data distribution
    distribution_stats = {}
    for client_id, indices in client_idxs.items():
        client_labels = [labels[idx] for idx in indices]
        unique_labels, counts = np.unique(client_labels, return_counts=True)
        distribution_stats[client_id] = {int(label): int(count) for label, count in zip(unique_labels, counts)}
    
    return client_idxs, distribution_stats

def save_figure(fig, base_path):
    """Save figure in both PNG and SVG formats"""
    # Save PNG
    png_path = f"{base_path}.png"
    fig.savefig(png_path, bbox_inches='tight', dpi=150)
    print(f"PNG saved to: {png_path}")
    
    # Save SVG
    svg_path = f"{base_path}.svg"
    fig.savefig(svg_path, bbox_inches='tight', format='svg')
    print(f"SVG saved to: {svg_path}")
    
    return png_path, svg_path

def visualize_distribution(distribution_stats, num_clients, num_classes, dataset_name, alpha, output_dir="./figures"):
    """Visualize client data distribution with separate figures, with clients sorted by data amount"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Data processing
    client_ids = list(distribution_stats.keys())
    
    # Create complete data matrix
    data_matrix = np.zeros((num_clients, num_classes))
    for client_id in client_ids:
        for class_id in range(num_classes):
            data_matrix[client_id, class_id] = distribution_stats[client_id].get(class_id, 0)
    
    # Calculate total samples per client for sorting
    client_totals = data_matrix.sum(axis=1)
    
    # Get sorted indices by total data amount (descending order)
    sorted_indices = np.argsort(-client_totals)
    
    # Create sorted version of data matrix
    sorted_data_matrix = data_matrix[sorted_indices]
    
    # Create sorted client IDs
    sorted_client_ids = [client_ids[i] for i in sorted_indices]
    
    # Create original to sorted index mapping for labels
    original_to_sorted = {client_ids[i]: f"Client {client_ids[i]}\n({int(client_totals[i])})" for i in range(num_clients)}
    sorted_client_labels = [original_to_sorted[id] for id in sorted_client_ids]
    
    # Prepare color map
    cmap = plt.cm.get_cmap('tab10', num_classes)
    colors = [cmap(i) for i in range(num_classes)]
    
    # 1. Stacked bar chart - Shows total samples and class distribution per client (SORTED)
    plt.figure(figsize=(12, 6), dpi=150)
    bottom = np.zeros(num_clients)
    for class_id in range(num_classes):
        class_counts = sorted_data_matrix[:, class_id]
        plt.bar(range(num_clients), class_counts, bottom=bottom, color=colors[class_id], 
               label=f'Class {class_id}', edgecolor='white', width=0.8)
        bottom += class_counts
    
    plt.title(f'{dataset_name} Data Distribution (alpha={alpha}, Sorted)', fontsize=14)
    plt.xlabel('Client ID (sorted by total data amount)', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(range(num_clients), [f"{sorted_client_ids[i]}" for i in range(num_clients)])
    plt.legend(title="Data Classes", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save sorted bar chart
    sorted_bar_base = os.path.join(output_dir, f"{dataset_name}_distribution_bar_sorted_alpha{alpha}")
    sorted_bar_png, sorted_bar_svg = save_figure(plt.gcf(), sorted_bar_base)
    plt.close()
    
    # Regular unsorted charts (original logic)
    # 1. Stacked bar chart - Shows total samples and class distribution per client
    plt.figure(figsize=(10, 6), dpi=150)
    bottom = np.zeros(num_clients)
    for class_id in range(num_classes):
        class_counts = data_matrix[:, class_id]
        plt.bar(client_ids, class_counts, bottom=bottom, color=colors[class_id], 
               label=f'Class {class_id}', edgecolor='white', width=0.8)
        bottom += class_counts
    
    plt.title(f'{dataset_name} Data Distribution (alpha={alpha})', fontsize=14)
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Number of Samples', fontsize=12)
    plt.xticks(client_ids)
    plt.legend(title="Data Classes", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save bar chart
    bar_base = os.path.join(output_dir, f"{dataset_name}_distribution_bar_alpha{alpha}")
    bar_png, bar_svg = save_figure(plt.gcf(), bar_base)
    plt.close()
    
    # 2. Heatmap - Shows data distribution matrix
    plt.figure(figsize=(10, 8), dpi=150)
    sns.heatmap(data_matrix, annot=True, fmt='g', cmap='viridis',
               xticklabels=[f'Class {i}' for i in range(num_classes)],
               yticklabels=[f'Client {i}' for i in client_ids],
               cbar_kws={"shrink": 0.8})
    
    plt.title(f'{dataset_name} Distribution Heatmap (alpha={alpha})', fontsize=14)
    plt.xlabel('Data Classes', fontsize=12)
    plt.ylabel('Client ID', fontsize=12)
    plt.tight_layout()
    
    # Save heatmap
    heatmap_base = os.path.join(output_dir, f"{dataset_name}_distribution_heatmap_alpha{alpha}")
    heatmap_png, heatmap_svg = save_figure(plt.gcf(), heatmap_base)
    plt.close()
    
    # 3. Stacked percentage bar chart - Shows class proportions per client
    plt.figure(figsize=(10, 6), dpi=150)
    data_percentage = data_matrix / data_matrix.sum(axis=1, keepdims=True) * 100
    
    bottom_percentage = np.zeros(num_clients)
    for class_id in range(num_classes):
        class_percentage = data_percentage[:, class_id]
        plt.bar(client_ids, class_percentage, bottom=bottom_percentage, color=colors[class_id], 
               label=f'Class {class_id}', edgecolor='white', width=0.8)
        bottom_percentage += class_percentage
    
    plt.title(f'{dataset_name} Percentage Distribution (alpha={alpha})', fontsize=14)
    plt.xlabel('Client ID', fontsize=12)
    plt.ylabel('Percentage (%)', fontsize=12)
    plt.xticks(client_ids)
    plt.ylim(0, 100)
    plt.legend(title="Data Classes", bbox_to_anchor=(1.01, 1), loc='upper left')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    # Save percentage bar chart
    percentage_base = os.path.join(output_dir, f"{dataset_name}_distribution_percentage_alpha{alpha}")
    percentage_png, percentage_svg = save_figure(plt.gcf(), percentage_base)
    plt.close()
    
    # Also create the combined figure (for backward compatibility)
    fig, axes = plt.subplots(1, 3, figsize=(24, 8), dpi=150)
    
    # 1. Stacked bar chart (SORTED)
    bottom = np.zeros(num_clients)
    for class_id in range(num_classes):
        class_counts = sorted_data_matrix[:, class_id]
        axes[0].bar(range(num_clients), class_counts, bottom=bottom, color=colors[class_id], 
                  label=f'Class {class_id}', edgecolor='white', width=0.8)
        bottom += class_counts
    
    axes[0].set_title(f'{dataset_name} Data Distribution (alpha={alpha}, Sorted)', fontsize=14)
    axes[0].set_xlabel('Client ID (sorted by total data amount)', fontsize=12)
    axes[0].set_ylabel('Number of Samples', fontsize=12)
    axes[0].set_xticks(range(num_clients))
    axes[0].set_xticklabels([f"{sorted_client_ids[i]}" for i in range(num_clients)])
    axes[0].legend(title="Data Classes", bbox_to_anchor=(1.01, 1), loc='upper left')
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)
    
    # 2. Heatmap
    sns.heatmap(data_matrix, annot=True, fmt='g', cmap='viridis',
               xticklabels=[f'Class {i}' for i in range(num_classes)],
               yticklabels=[f'Client {i}' for i in client_ids],
               ax=axes[1], cbar_kws={"shrink": 0.8})
    
    axes[1].set_title(f'{dataset_name} Distribution Heatmap (alpha={alpha})', fontsize=14)
    axes[1].set_xlabel('Data Classes', fontsize=12)
    axes[1].set_ylabel('Client ID', fontsize=12)
    
    # 3. Stacked percentage bar chart
    data_percentage = data_matrix / data_matrix.sum(axis=1, keepdims=True) * 100
    bottom_percentage = np.zeros(num_clients)
    for class_id in range(num_classes):
        class_percentage = data_percentage[:, class_id]
        axes[2].bar(client_ids, class_percentage, bottom=bottom_percentage, color=colors[class_id], 
                  label=f'Class {class_id}', edgecolor='white', width=0.8)
        bottom_percentage += class_percentage
    
    axes[2].set_title(f'{dataset_name} Percentage Distribution (alpha={alpha})', fontsize=14)
    axes[2].set_xlabel('Client ID', fontsize=12)
    axes[2].set_ylabel('Percentage (%)', fontsize=12)
    axes[2].set_xticks(client_ids)
    axes[2].set_ylim(0, 100)
    axes[2].legend(title="Data Classes", bbox_to_anchor=(1.01, 1), loc='upper left')
    axes[2].grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    # Save combined figure
    combined_base = os.path.join(output_dir, f"{dataset_name}_distribution_combined_alpha{alpha}")
    combined_png, combined_svg = save_figure(plt.gcf(), combined_base)
    plt.close()
    
    # Save data statistics
    stats_path = os.path.join(output_dir, f"{dataset_name}_distribution_stats_alpha{alpha}.json")
    with open(stats_path, 'w') as f:
        json.dump(distribution_stats, f, indent=4)
    
    print(f"Distribution statistics saved to: {stats_path}")
    
    return {
        'bar_chart': {'png': bar_png, 'svg': bar_svg},
        'sorted_bar_chart': {'png': sorted_bar_png, 'svg': sorted_bar_svg},
        'heatmap': {'png': heatmap_png, 'svg': heatmap_svg},
        'percentage_chart': {'png': percentage_png, 'svg': percentage_svg},
        'combined': {'png': combined_png, 'svg': combined_svg},
        'stats': stats_path
    }

def main():
    parser = argparse.ArgumentParser(description='Visualize Federated Learning Data Distribution')
    parser.add_argument('--dataset', type=str, default='CIFAR10', 
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10'],
                        help='Dataset name (default: CIFAR10)')
    parser.add_argument('--num_clients', type=int, default=10, 
                        help='Number of clients (default: 10)')
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='Dirichlet distribution alpha parameter (default: 0.1)')
    parser.add_argument('--output_dir', type=str, default='./figures', 
                        help='Output directory for figures (default: ./figures)')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Load dataset
    train_dataset, classes = get_dataset(args.dataset)
    num_classes = len(classes)
    
    # Distribute data
    client_idxs, distribution_stats = distribute_data_noniid_label(
        train_dataset, args.num_clients, args.alpha, args.seed
    )
    
    # Visualize data distribution
    output_paths = visualize_distribution(
        distribution_stats, args.num_clients, num_classes, 
        args.dataset, args.alpha, args.output_dir
    )
    
    print(f"Visualization complete!")
    print(f"Generated files:")
    for key, path_dict in output_paths.items():
        print(f"- {key}:")
        if isinstance(path_dict, dict):
            for format, file_path in path_dict.items():
                print(f"  - {format}: {file_path}")
        else:
            print(f"  - {path_dict}")

if __name__ == "__main__":
    main() 