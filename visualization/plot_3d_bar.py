import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse
import matplotlib.colors as mcolors

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Create 3D bar charts for federated learning metrics')
    parser.add_argument('--metrics_dir', type=str, default='./figures/csv',
                       help='Directory containing metrics CSV files')
    parser.add_argument('--output_dir', type=str, default='./figures/3d_plots',
                       help='Directory to save output charts')
    parser.add_argument('--metrics', type=str, choices=['accuracy', 'f1'], 
                        default='accuracy', help='Metrics to plot')
    parser.add_argument('--dpi', type=int, default=300,
                       help='DPI for saved images')
    parser.add_argument('--card_width', type=float, default=0.5,
                       help='Bar chart width')
    parser.add_argument('--card_depth', type=float, default=0.05,
                       help='Bar chart depth')
    parser.add_argument('--alpha', type=float, default=0.7,
                       help='Bar chart transparency')
    return parser.parse_args()

def load_metrics(metrics_dir):
    """Load metrics data"""
    # Try to load the integrated metrics file
    metrics_file = os.path.join(metrics_dir, 'all_metrics.csv')
    if os.path.exists(metrics_file):
        print(f"Loading metrics file: {metrics_file}")
        df = pd.read_csv(metrics_file)
        return df
    
    # If the integrated file doesn't exist, try loading individual files for each dataset
    datasets = ['MNIST', 'FashionMNIST', 'CIFAR10']
    all_data = []
    
    for dataset in datasets:
        dataset_file = os.path.join(metrics_dir, f'{dataset}_metrics.csv')
        if os.path.exists(dataset_file):
            dataset_df = pd.read_csv(dataset_file)
            all_data.append(dataset_df)
    
    if all_data:
        return pd.concat(all_data, ignore_index=True)
    
    raise FileNotFoundError(f"No metrics files found in {metrics_dir}")

def create_distribution_3d_plot(df, metric_name, output_dir, dpi=300, card_width=0.3, card_depth=0.1, bar_alpha=0.9):
    """Create 3D charts showing performance across different distribution types (alpha values)"""
    # Convert numeric Alpha to float
    df['Alpha'] = df['Alpha'].apply(lambda x: float(x) if x != 'iid' else x)
    
    # Exclude specific algorithms - exclude algorithms without results or with issues
    algorithms_to_exclude = ['MultiKrum', 'Bulyan', 'FedNova']
    df = df[~df['Algorithm'].isin(algorithms_to_exclude)]
    
    # Get all datasets
    datasets = df['Dataset'].unique()
    if len(datasets) != 3:
        print(f"Warning: Expected 3 datasets, but found {len(datasets)}")
    
    # Ensure dataset order
    ordered_datasets = []
    for dataset_name in ['MNIST', 'FashionMNIST', 'CIFAR10']:
        if dataset_name in datasets:
            ordered_datasets.append(dataset_name)
    datasets = ordered_datasets
    
    # Get unique Alpha values and sort them
    alpha_values = df['Alpha'].unique()
    numeric_alphas = [a for a in alpha_values if a != 'iid']
    numeric_alphas.sort()
    ordered_alphas = numeric_alphas.copy()
    if 'iid' in alpha_values:
        ordered_alphas.append('iid')
    
    # Color mapping - ensure consistency with colors in the figures
    color_map = {
        'FedAvg': '#FF3300',        # Red
        'FedProx': '#FF9900',       # Orange
        'SCAFFOLD': '#0066FF',      # Blue
        'Median': '#FFCC00',        # Yellow
        'TrimmedMean': '#3333FF',   # Dark Blue
        'Krum': '#00CC66',          # Green
        'Auror': '#6600CC',         # Purple
        'FedCVG': '#66CCFF',        # Light Blue
        'SignSGD': '#33CC33',       # Green
        'GeoMed': '#CC6600'         # Orange-Brown
    }
    
    # Set different z-axis ranges for each dataset
    z_ranges = {
        'MNIST': (95, 100),
        'FashionMNIST': (70, 100),
        'CIFAR10': (0, 100)
    }
    
    # Set different offsets for different datasets
    offsets = {
        'MNIST': {'x': 0, 'y': 0},
        'FashionMNIST': {'x': 0, 'y': 0},
        'CIFAR10': {'x': 0.0, 'y': 0.0}
    }
    
    # Determine which metric column to use
    metric_column = f'Max_{metric_name.capitalize()}'
    
    # Create chart
    fig = plt.figure(figsize=(20, 8), dpi=dpi)  # Increase chart width
    
    # Process each dataset
    for idx, dataset in enumerate(datasets):
        ax = fig.add_subplot(131 + idx, projection='3d')
        
        # Get available algorithms for current dataset
        dataset_df = df[df['Dataset'] == dataset]
        available_algos = dataset_df['Algorithm'].unique().tolist()
        available_algos.sort()  # Sort algorithm names
        
        # Get coordinate offset for current dataset
        x_offset = offsets[dataset]['x']
        y_offset = offsets[dataset]['y']
        
        # Get z-axis range
        z_min, z_max = z_ranges[dataset]
        
        # Prepare plotting data
        plot_data = []
        for i, alpha in enumerate(ordered_alphas):
            for j, algo in enumerate(available_algos):
                row = dataset_df[(dataset_df['Algorithm'] == algo) & 
                                (dataset_df['Alpha'] == alpha)]
                
                if not row.empty:
                    value = row[metric_column].values[0]
                    # Assign colors for any algorithms without a color mapping
                    if algo not in color_map:
                        color_map[algo] = plt.cm.tab20(len(color_map) % 20)
                    
                    plot_data.append({
                        'Alpha_idx': i + x_offset,
                        'Algorithm_idx': j + y_offset,
                        'Value': value,
                        'Height': value - z_min,  # Bar height is value minus z-axis minimum
                        'Color': color_map.get(algo, color_map[algo])
                    })
                else:
                    print(f"Missing data: {dataset}, {algo}, alpha={alpha}")
                    plot_data.append({
                        'Alpha_idx': i + x_offset,
                        'Algorithm_idx': j + y_offset,
                        'Value': 0,
                        'Height': 0,  # If value is 0, height is also 0
                        'Color': color_map.get(algo, plt.cm.tab20(len(color_map) % 20))
                    })
        
        # Draw 3D bar chart
        for item in plot_data:
            x = item['Alpha_idx']
            y = item['Algorithm_idx']
            z = z_min  # Start from z-axis minimum
            dx = card_width
            dy = card_depth
            dz = item['Height']  # Use calculated height
            color = item['Color']
            
            # Only draw bars with height
            if dz > 0:
                ax.bar3d(x, y, z, dx, dy, dz, color=color, shade=True, alpha=bar_alpha, edgecolor='none')
        
        # Set coordinate axis style
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False
        
        # Adjust viewing angle for better visual effect - maintain consistent angle
        ax.view_init(elev=30, azim=-60)
        
        # Adjust coordinate axis range to ensure all bars are in visible area
        ax.set_xlim(x_offset-0.2, len(ordered_alphas) + x_offset + 0.2)
        ax.set_ylim(y_offset-0.2, len(available_algos) + y_offset + 0.2)
        
        # Set coordinate axis labels
        ax.set_xticks(np.arange(len(ordered_alphas)) + x_offset + dx/2)
        x_labels = []
        for alpha in ordered_alphas:
            if alpha == 'iid':
                x_labels.append('IID')
            else:
                x_labels.append(f'α={alpha}')
        ax.set_xticklabels(x_labels)
        
        # Set z-axis range to the specific range for this dataset
        ax.set_zlim(z_min, z_max)
        
        # Set labels
        ax.set_zlabel(f'{metric_name.capitalize()} (%)', fontsize=12)
    
    # Add global legend - only keep right side legend
    legend_handles = []
    legend_labels = []
    
    # Get all algorithms and sort them
    all_algorithms = list(set(algo for dataset in datasets for algo in df[df['Dataset'] == dataset]['Algorithm'].unique()))
    all_algorithms.sort()
    
    for algo in all_algorithms:
        if algo in color_map:
            color = color_map[algo]
            legend_handles.append(plt.Rectangle((0, 0), 1, 1, color=color))
            legend_labels.append(algo)
    
    fig.legend(
        legend_handles, 
        legend_labels, 
        loc='upper right',
        bbox_to_anchor=(0.98, 0.95),
        title="Algorithms"
    )
    
    # Adjust layout - increase left margin
    plt.tight_layout(rect=[0.02, 0, 0.95, 0.95])
    
    # Save chart
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{metric_name}_distribution_comparison.png')
    plt.savefig(output_path, bbox_inches='tight', dpi=dpi)
    
    # Save SVG format
    svg_path = os.path.join(output_dir, f'{metric_name}_distribution_comparison.svg')
    plt.savefig(svg_path, bbox_inches='tight', format='svg')
    
    print(f"{metric_name.capitalize()} 3D comparison chart across different distribution types saved to: {output_path}")
    plt.close(fig)

def main():
    """Main function"""
    args = parse_args()
    
    print(f"Loading metrics from {args.metrics_dir}")
    print(f"Output will be saved to {args.output_dir}")
    
    # Load metrics data
    try:
        metrics_df = load_metrics(args.metrics_dir)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create 3D charts
    create_distribution_3d_plot(
        metrics_df,
        args.metrics,
        args.output_dir,
        args.dpi,
        args.card_width,
        args.card_depth,
        args.alpha
    )
    
    print(f"\n3D chart generation complete. Files saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 