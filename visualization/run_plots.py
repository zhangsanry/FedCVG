import os
import argparse
import subprocess

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Run Federated Learning Visualization Scripts')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='Results file directory')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Output directory')
    parser.add_argument('--select_algorithms', type=str, nargs='*',
                        help='Specify algorithms to display (shows all available if not specified)')
    parser.add_argument('--max_algorithms', type=int, default=0,
                        help='Maximum number of algorithms to display per chart (0 means no limit)')
    parser.add_argument('--dpi', type=int, default=300,
                        help='Image DPI')
    parser.add_argument('--skip_model_results', action='store_true',
                        help='Skip model results visualization')
    parser.add_argument('--skip_metrics_extraction', action='store_true',
                        help='Skip metrics extraction')
    parser.add_argument('--skip_3d_plots', action='store_true',
                        help='Skip 3D chart generation')
    return parser.parse_args()

def run_command(cmd, desc):
    """Run command and print output"""
    print(f"\n{'='*80}")
    print(f"Executing: {desc}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'-'*80}")
    
    try:
        subprocess.run(cmd, check=True)
        print(f"{desc} completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"Error: {desc} failed with exit code {e.returncode}")
        print(f"Error message: {e}")

def main():
    """Main function"""
    args = parse_args()
    
    # Ensure directories exist
    os.makedirs(args.output_dir, exist_ok=True)
    csv_dir = os.path.join(args.output_dir, 'csv')
    os.makedirs(csv_dir, exist_ok=True)
    plots_3d_dir = os.path.join(args.output_dir, '3d_plots')
    os.makedirs(plots_3d_dir, exist_ok=True)
    
    # Create command line commands
    model_results_cmd = [
        'python', 'visualization/visualize_model_results.py',
        '--results_dir', args.results_dir,
        '--output_dir', args.output_dir,
        '--dpi', str(args.dpi)
    ]
    
    if args.select_algorithms:
        model_results_cmd.extend(['--select_algorithms'] + args.select_algorithms)
    
    if args.max_algorithms > 0:
        model_results_cmd.extend(['--max_algorithms', str(args.max_algorithms)])
    
    extract_metrics_cmd = [
        'python', 'visualization/extract_metrics.py',
        '--results_dir', args.results_dir,
        '--output_dir', csv_dir
    ]
    
    plot_3d_cmd = [
        'python', 'visualization/plot_3d_bar.py',
        '--metrics_dir', csv_dir,
        '--output_dir', plots_3d_dir,
        '--metrics', 'both',
        '--dpi', str(args.dpi)
    ]
    
    # Execute visualization steps
    if not args.skip_model_results:
        run_command(model_results_cmd, "Model Results Visualization")
    else:
        print("Skipping model results visualization")
    
    if not args.skip_metrics_extraction:
        run_command(extract_metrics_cmd, "Metrics Extraction")
    else:
        print("Skipping metrics extraction")
    
    if not args.skip_3d_plots:
        run_command(plot_3d_cmd, "3D Chart Generation")
    else:
        print("Skipping 3D chart generation")
    
    print("\nVisualization process completed!")
    print(f"All output files have been saved to: {args.output_dir}")

if __name__ == "__main__":
    main() 