import os
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
import seaborn as sns
from pathlib import Path
import re

# Define lists of datasets and alpha values
DATASETS = ['MNIST_LeNet', 'FashionMNIST_LeNet', 'CIFAR10_ResNet34']
ALPHAS = [0.1, 0.5, 1.0]
DISTRIBUTION_TYPES = ALPHAS + ['iid']  # Include both IID and Non-IID

# Color mapping, assign fixed colors to algorithms for consistency in visualization
COLOR_MAP = {
    'FedAvg': 'gray',
    'Krum': 'gold',
    'MultiKrum': 'orange',
    'SCAFFOLD': 'green',
    'Auror': 'red',
    'FedProx': 'blue',
    'Median': 'purple',
    'TrimmedMean': 'brown',
    'Bulyan': 'pink',
    'SignGuard': 'cyan',
    'RLR': 'magenta',
    'FABA': 'lime',
    'FLTrust': 'teal',
    'Dnc': 'salmon',
    'SignFlip': 'darkkhaki',
    'FedCVG_SCAFFOLD_Cluster_Rep_GradMem': '#FF1493',  # Hot pink
    'FedCVG_FedProx': '#00FFFF'    # Cyan
}

# Main algorithms to compare (displayed in legend order, used for sorting only)
MAIN_ALGORITHMS = ['FedAvg', 'Krum', 'SCAFFOLD', 'Auror', 'FedProx', 'Median', 'TrimmedMean', 'Bulyan', 'MultiKrum', 'FedCVG_SCAFFOLD_Cluster_Rep_GradMem', 'FedCVG_SCAFFOLD', 'FedCVG_FedProx']

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Visualize Federated Learning Model Results')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing results files')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Directory to save output figures')
    parser.add_argument('--select_algorithms', type=str, nargs='+', 
                       help='Only show these specific algorithms (optional)')
    parser.add_argument('--max_algorithms', type=int, default=0,
                       help='Maximum number of algorithms to display (0 means all)')
    return parser.parse_args()

def load_result_file(file_path):
    """Read data from result file"""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            data = json.loads(content)
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading {file_path}: {e}")
        return None

def get_available_algorithms(results_dir, dataset, alpha):
    """Get list of available algorithms for specified dataset and alpha value"""
    dataset_dir = os.path.join(results_dir, dataset)
    if not os.path.exists(dataset_dir):
        return []
    
    algorithms = []
    
    if alpha == 'iid':
        # IID mode
        pattern = rf"\[(.*?)_{dataset.split('_')[1]}_iid_1234\]"
        
        for file_name in os.listdir(dataset_dir):
            # Try to match IID pattern
            match = re.match(pattern, file_name)
            if match:
                algo_name = match.group(1)
                algorithms.append(algo_name)
    else:
        # Regular algorithm pattern
        pattern = rf"\[(.*?)_{dataset.split('_')[1]}_non_iid_label_alpha{alpha}_1234\]"
        # FedCVG special pattern (originally FedHyb)
        fedhyb_pattern = rf"\[(FedHyb.*?)_{dataset.split('_')[1]}_non_iid_label_alpha{alpha}_1234\]"
        
        for file_name in os.listdir(dataset_dir):
            # Try to match FedCVG special pattern
            match = re.match(fedhyb_pattern, file_name)
            if match:
                # Rename FedHyb to FedCVG
                algo_name = match.group(1).replace("FedHyb", "FedCVG")
                algorithms.append(algo_name)
                continue
                
            # Try to match regular pattern
            match = re.match(pattern, file_name)
            if match:
                algo_name = match.group(1)
                algorithms.append(algo_name)
    
    return algorithms

def create_accuracy_plot(dataset, alpha, results_dir, output_dir, select_algorithms=None, max_algorithms=0):
    """Create accuracy curve plot for specific dataset and alpha value"""
    # Set larger figure size to accommodate more algorithms
    plt.figure(figsize=(12, 7), dpi=150)
    
    # Set chart style
    sns.set_style("whitegrid")
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Get available algorithms for current dataset and alpha value
    available_algorithms = get_available_algorithms(results_dir, dataset, alpha)
    
    if not available_algorithms:
        plt.close()
        print(f"No available algorithms for {dataset} with alpha={alpha}")
        return False
    
    # If specific algorithms are specified, only keep those
    if select_algorithms:
        algorithms_to_show = [algo for algo in available_algorithms if algo in select_algorithms]
        if not algorithms_to_show:
            print(f"None of the specified algorithms {select_algorithms} are available for {dataset} with alpha={alpha}")
            print(f"Available algorithms: {available_algorithms}")
            algorithms_to_show = available_algorithms
    else:
        # By default, show all available algorithms
        algorithms_to_show = available_algorithms
    
    # If maximum number of algorithms is specified, limit the number of displayed algorithms
    if max_algorithms > 0 and len(algorithms_to_show) > max_algorithms:
        # Prioritize main algorithms
        main_algos = [algo for algo in MAIN_ALGORITHMS if algo in algorithms_to_show][:max_algorithms]
        if len(main_algos) < max_algorithms:
            # Add other algorithms until maximum number is reached
            other_algos = [algo for algo in algorithms_to_show if algo not in main_algos]
            algorithms_to_show = main_algos + other_algos[:max_algorithms - len(main_algos)]
        else:
            algorithms_to_show = main_algos[:max_algorithms]
    
    max_rounds = 0
    valid_results = False
    dataset_dir = os.path.join(results_dir, dataset)
    
    # Sort algorithms to maintain consistent legend order
    def sort_key(algo):
        try:
            return MAIN_ALGORITHMS.index(algo)
        except ValueError:
            # For algorithms not in the main algorithm list, if it starts with FedCVG, give it higher priority
            if algo.startswith('FedCVG'):
                return len(MAIN_ALGORITHMS) - 0.5  # Give FedCVG algorithms higher priority
            # Other algorithms are sorted by their order in algorithms_to_show
            return len(MAIN_ALGORITHMS) + 1
    
    algorithms_to_show.sort(key=sort_key)
    
    # Assign a line style for each algorithm to make them easier to distinguish when there are many
    linestyles = ['-', '--', '-.', ':']
    
    # Mark FedCVG as already added to the legend
    fedcvg_added = False
    
    for i, algo_name in enumerate(algorithms_to_show):
        # Build file path
        if alpha == 'iid':
            # IID data format
            result_file = f"[{algo_name}_{dataset.split('_')[1]}_iid_1234]"
        elif 'FedCVG' in algo_name:
            # Convert FedCVG back to original filename FedHyb
            result_file = f"[{algo_name.replace('FedCVG', 'FedHyb')}_{dataset.split('_')[1]}_non_iid_label_alpha{alpha}_1234]"
        else:
            result_file = f"[{algo_name}_{dataset.split('_')[1]}_non_iid_label_alpha{alpha}_1234]"
        result_path = os.path.join(dataset_dir, result_file)
        
        # Read result data
        data = load_result_file(result_path)
        if data and 'server' in data and 'accuracy' in data['server']:
            accuracy = data['server']['accuracy']
            
            # Multiply accuracy values by 100 to convert to percentage
            accuracy = [acc * 100 for acc in accuracy]
            
            rounds = list(range(len(accuracy)))
            max_rounds = max(max_rounds, len(rounds))
            
            # Get algorithm color, use random color if not defined
            color = COLOR_MAP.get(algo_name, None)
            if color is None:
                # Assign color for undefined algorithms
                color_idx = i % len(sns.color_palette("husl", len(algorithms_to_show)))
                color = sns.color_palette("husl", len(algorithms_to_show))[color_idx]
            
            # Select line style, use solid line for FedCVG algorithms
            if 'FedCVG' in algo_name:
                linestyle = '-'
                linewidth = 3  # Use thicker lines to highlight
                # Simplify FedCVG display name and avoid duplication
                if fedcvg_added:
                    # If FedCVG has already been added, don't show label
                    display_name = '_nolegend_'
                else:
                    display_name = 'FedCVG'
                    fedcvg_added = True
            else:
                linestyle = linestyles[i % len(linestyles)]
                linewidth = 2
                display_name = algo_name
            
            # Plot accuracy curve
            plt.plot(rounds, accuracy, 
                     label=display_name, 
                     color=color,
                     linestyle=linestyle,
                     linewidth=linewidth)
            valid_results = True
    
    if not valid_results:
        plt.close()
        print(f"No valid results found for {dataset} with alpha={alpha}")
        return False
    
    # Set chart labels
    plt.xlabel('Communication Round', fontsize=12)
    plt.ylabel('Accuracy (%)', fontsize=12)
    
    # Set y-axis range based on dataset and alpha value
    if dataset.startswith('CIFAR10'):
        if alpha == 0.1:
            # CIFAR10, show 20-60% when α=0.1
            plt.ylim(20, 60)
        elif alpha == 0.5:
            # CIFAR10, show 55-80% when α=0.5
            plt.ylim(55, 80)
        else:  # alpha == 1.0
            # CIFAR10, show 70-85% when α=1.0
            plt.ylim(70, 85)
    elif dataset.startswith('FashionMNIST'):
        if alpha == 0.1:
            # FashionMNIST, show 60-90% when α=0.1
            plt.ylim(60, 90)
        elif alpha == 0.5:
            # FashionMNIST, show 75-90% when α=0.5
            plt.ylim(75, 90)
        else:  # alpha == 1.0
            # FashionMNIST, show 80-90% when α=1.0
            plt.ylim(80, 90)
    else:  # MNIST
        if alpha == 1.0:
            # MNIST, show 97.5-100% when α=1.0
            plt.ylim(97.5, 100)
        elif alpha == 0.5:
            # MNIST, show 95-100% when α=0.5
            plt.ylim(95, 100)
        else:  # alpha == 0.1
            # MNIST, show 80-100% when α=0.1
            plt.ylim(80, 100)
    
    # Set x-axis from 0 to 100, with intervals of 10
    max_x = min(100, max_rounds if max_rounds > 0 else 100)
    plt.xlim(0, max_x)
    plt.xticks(np.arange(0, max_x+1, 10))
    
    # Add legend, place in the upper right corner inside the chart, use two-column layout
    plt.legend(loc='upper right', frameon=True, ncol=2)
    
    # Create PNG and SVG output directories
    png_dir = os.path.join(output_dir, 'png')
    svg_dir = os.path.join(output_dir, 'svg')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    # Save PNG format
    png_path = os.path.join(png_dir, f"{dataset.split('_')[0]}_alpha{alpha}_accuracy.png")
    plt.tight_layout()
    plt.savefig(png_path, bbox_inches='tight')
    
    # Save SVG format
    svg_path = os.path.join(svg_dir, f"{dataset.split('_')[0]}_alpha{alpha}_accuracy.svg")
    plt.savefig(svg_path, bbox_inches='tight', format='svg')
    
    plt.close()
    
    print(f"Saved accuracy plot for {dataset} with alpha={alpha}")
    print(f" - Algorithms shown: {algorithms_to_show}")
    print(f" - PNG: {png_path}")
    print(f" - SVG: {svg_path}")
    
    return True

def main():
    """Main function"""
    args = parse_args()
    
    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Check if results directory exists
    if not os.path.exists(args.results_dir):
        print(f"Error: Results directory '{args.results_dir}' does not exist.")
        return
    
    # Get all dataset folders in the results directory
    available_datasets = []
    for dataset in DATASETS:
        if os.path.exists(os.path.join(args.results_dir, dataset)):
            available_datasets.append(dataset)
    
    if not available_datasets:
        print(f"Error: No dataset folders found in '{args.results_dir}'.")
        return
    
    # Create dictionary to store visualization results
    successful_plots = {dataset: [] for dataset in DATASETS}
    
    # Create accuracy curve plots for each dataset and alpha value
    for dataset in available_datasets:
        for alpha in DISTRIBUTION_TYPES:  # Use list that includes IID
            success = create_accuracy_plot(
                dataset, alpha, args.results_dir, args.output_dir, 
                select_algorithms=args.select_algorithms,
                max_algorithms=args.max_algorithms
            )
            if success:
                successful_plots[dataset].append(alpha)
    
    # Output summary
    print("\nVisualization Complete!")
    print("=" * 50)
    print("Summary of generated plots:")
    
    for dataset, alphas in successful_plots.items():
        if dataset in available_datasets:
            if alphas:
                print(f"- {dataset}: Alpha values {', '.join(map(str, alphas))}")
            else:
                print(f"- {dataset}: No plots generated")
    
    print("=" * 50)
    print(f"All plots saved to: {args.output_dir}")
    print(f" - PNG files in: {os.path.join(args.output_dir, 'png')}")
    print(f" - SVG files in: {os.path.join(args.output_dir, 'svg')}")

if __name__ == "__main__":
    main() 