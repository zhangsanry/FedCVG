import os
import json
import pandas as pd
import argparse
import glob
import numpy as np

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Extract table data from experiment results')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Directory containing experiment results')
    parser.add_argument('--output_dir', type=str, default='./figures/table',
                        help='Directory for output CSV files')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['MNIST', 'FashionMNIST', 'CIFAR10', 'Cora', 'Citeseer', 'Pubmed'],
                        help='List of datasets to process')
    parser.add_argument('--last_rounds', type=int, default=50,
                        help='Number of last rounds to use for calculating averages')
    return parser.parse_args()

def extract_metadata(file_name):
    """Extract metadata from filename"""
    # Remove brackets
    file_name = file_name.replace('[', '').replace(']', '')
    
    # Split filename
    parts = file_name.split('_')
    
    # Extract algorithm name
    algorithm = parts[0]
    
    # Extract alpha value
    alpha = None
    for part in parts:
        if part.startswith('alpha'):
            alpha = float(part[5:])
            break
    
    # Extract dataset name
    dataset = None
    for dataset_name in ['MNIST', 'FashionMNIST', 'CIFAR10', 'Cora', 'Citeseer', 'Pubmed']:
        if dataset_name in file_name:
            dataset = dataset_name
            break
    
    return {
        'algorithm': algorithm,
        'dataset': dataset,
        'alpha': alpha
    }

def process_result_file(file_path, last_rounds=50):
    """Process a single result file and calculate statistics"""
    try:
        # Read file content
        with open(file_path, 'r') as f:
            content = f.read()
        
        # Parse JSON
        data = json.loads(content)
        server_data = data.get('server', {})
        
        # Extract accuracy and F1 score
        accuracy = server_data.get('accuracy', [])
        f1_score = server_data.get('f1_score', [])
        
        # Check if data is sufficient
        if len(accuracy) < last_rounds:
            print(f"Warning: Data in {file_path} is less than {last_rounds} rounds")
            return None
        
        # Calculate average and range for the last few rounds, remove one highest and one lowest value
        last_accuracy = accuracy[-last_rounds:]
        # Sort array
        sorted_accuracy = sorted(last_accuracy)
        # Remove highest and lowest values
        trimmed_accuracy = sorted_accuracy[1:-1]
        
        # Calculate average using trimmed data
        acc_mean = np.mean(trimmed_accuracy) * 100  # Convert to percentage
        # Calculate max and min of trimmed data
        acc_max = np.max(trimmed_accuracy) * 100
        acc_min = np.min(trimmed_accuracy) * 100
        
        # Calculate maximum deviation from the mean
        max_diff = max(acc_max - acc_mean, acc_mean - acc_min)
        
        if f1_score and len(f1_score) >= last_rounds:
            last_f1 = f1_score[-last_rounds:]
            # Sort array
            sorted_f1 = sorted(last_f1)
            # Remove highest and lowest values
            trimmed_f1 = sorted_f1[1:-1]
            
            # Calculate average and range using trimmed data
            f1_mean = np.mean(trimmed_f1) * 100
            f1_max = np.max(trimmed_f1) * 100
            f1_min = np.min(trimmed_f1) * 100
            # Calculate maximum deviation from the mean
            f1_max_diff = max(f1_max - f1_mean, f1_mean - f1_min)
        else:
            f1_mean = None
            f1_max_diff = None
        
        # Extract metadata
        metadata = extract_metadata(os.path.basename(file_path))
        
        result = {
            'algorithm': metadata['algorithm'],
            'dataset': metadata['dataset'],
            'alpha': metadata['alpha'],
            'acc_mean': acc_mean,
            'acc_range': max_diff,
            'f1_mean': f1_mean,
            'f1_range': f1_max_diff
        }
        
        return result
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None

def map_alpha_to_m(alpha):
    """Map alpha value to M value"""
    return int(alpha * 10)

def main():
    """Main function"""
    args = parse_args()
    
    print(f"Loading experiment results from {args.results_dir}")
    print(f"Using the last {args.last_rounds} rounds to calculate metrics")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_results = []
    
    # Process each dataset
    for dataset in args.datasets:
        # Find dataset directory
        dataset_dir = os.path.join(args.results_dir, f"{dataset}_*")
        dataset_dirs = glob.glob(dataset_dir)
        
        if not dataset_dirs:
            print(f"Warning: No result directory found for {dataset}")
            continue
        
        dataset_results = []
        
        # Process each dataset directory
        for dir_path in dataset_dirs:
            # Find all result files
            result_files = glob.glob(os.path.join(dir_path, "*"))
            
            for file_path in result_files:
                result = process_result_file(file_path, args.last_rounds)
                if result is not None:
                    dataset_results.append(result)
                    all_results.append(result)
        
        if dataset_results:
            # Create DataFrame
            df = pd.DataFrame(dataset_results)
            
            # Save as CSV
            output_path = os.path.join(args.output_dir, f"{dataset}_stats.csv")
            df.to_csv(output_path, index=False)
            print(f"{dataset} data saved to: {output_path}")
            
            # Create pivot table
            if not df.empty:
                # Create pivot table, grouped by algorithm and alpha
                pivot_df = df.pivot_table(
                    index='algorithm', 
                    columns='alpha',
                    values=['acc_mean', 'acc_range'],
                    aggfunc='first'
                )
                
                # Save pivot table
                pivot_path = os.path.join(args.output_dir, f"{dataset}_pivot.csv")
                pivot_df.to_csv(pivot_path)
                print(f"{dataset} pivot table saved to: {pivot_path}")
                
                # Create a formatted table, suitable for table display
                formatted_data = []
                methods = df['algorithm'].unique()
                alphas = sorted(df['alpha'].unique())
                
                for method in methods:
                    row = {'Method': method}
                    for alpha in alphas:
                        subset = df[(df['algorithm'] == method) & (df['alpha'] == alpha)]
                        if not subset.empty:
                            m_value = map_alpha_to_m(alpha)
                            key = f"M={m_value}"
                            row[key] = f"{subset['acc_mean'].values[0]:.2f}±{subset['acc_range'].values[0]:.2f}"
                    formatted_data.append(row)
                
                formatted_df = pd.DataFrame(formatted_data)
                formatted_path = os.path.join(args.output_dir, f"{dataset}_formatted.csv")
                formatted_df.to_csv(formatted_path, index=False)
                print(f"{dataset} formatted table saved to: {formatted_path}")
    
    # Save merged data for all results
    if all_results:
        # Create DataFrame
        all_df = pd.DataFrame(all_results)
        
        # Save as CSV
        all_path = os.path.join(args.output_dir, "all_stats.csv")
        all_df.to_csv(all_path, index=False)
        print(f"All data merged and saved to: {all_path}")
        
        # Create comprehensive table, including all datasets and alpha values
        combined_data = []
        methods = all_df['algorithm'].unique()
        
        for method in methods:
            row = {'Method': method}
            for dataset in args.datasets:
                dataset_df = all_df[all_df['dataset'] == dataset]
                if not dataset_df.empty:
                    alphas = sorted(dataset_df['alpha'].unique())
                    for alpha in alphas:
                        subset = dataset_df[(dataset_df['algorithm'] == method) & 
                                        (dataset_df['alpha'] == alpha)]
                        if not subset.empty:
                            m_value = map_alpha_to_m(alpha)
                            key = f"{dataset}_M{m_value}"
                            row[key] = f"{subset['acc_mean'].values[0]:.2f}±{subset['acc_range'].values[0]:.2f}"
            combined_data.append(row)
        
        combined_df = pd.DataFrame(combined_data)
        combined_path = os.path.join(args.output_dir, "combined_table.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"Comprehensive table saved to: {combined_path}")
    
    print("\nData collection and processing complete.")

if __name__ == "__main__":
    main() 