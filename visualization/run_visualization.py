import os
import argparse
import subprocess

def main():
    """Run model results visualization script"""
    parser = argparse.ArgumentParser(description='Run Federated Learning Model Results Visualization')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='Directory containing results files')
    parser.add_argument('--output_dir', type=str, default='./figures',
                       help='Directory to save output figures')
    parser.add_argument('--select_algorithms', type=str, nargs='+',
                       help='Only show these specific algorithms (optional)')
    parser.add_argument('--max_algorithms', type=int, default=0,
                       help='Maximum number of algorithms to display (0 means all)')
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    # Create PNG and SVG subdirectories
    png_dir = os.path.join(args.output_dir, 'png')
    svg_dir = os.path.join(args.output_dir, 'svg')
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(svg_dir, exist_ok=True)
    
    print("=" * 60)
    print("Running Federated Learning Model Results Visualization")
    print("=" * 60)
    print(f"Results directory: {args.results_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"  - PNG files will be saved to: {png_dir}")
    print(f"  - SVG files will be saved to: {svg_dir}")
    
    if args.select_algorithms:
        print(f"Selected algorithms: {', '.join(args.select_algorithms)}")
    else:
        print("Showing all available algorithms")
        
    if args.max_algorithms > 0:
        print(f"Maximum algorithms to display: {args.max_algorithms}")
    else:
        print("No limit on number of algorithms displayed")
    
    print("=" * 60)
    
    # Build command
    cmd = f"python visualization/visualize_model_results.py --results_dir {args.results_dir} --output_dir {args.output_dir}"
    
    if args.select_algorithms:
        algorithms_param = " ".join([f'"{algo}"' for algo in args.select_algorithms])
        cmd += f" --select_algorithms {algorithms_param}"
    
    if args.max_algorithms > 0:
        cmd += f" --max_algorithms {args.max_algorithms}"
    
    # Run command
    print(f"Executing command: {cmd}")
    subprocess.run(cmd, shell=True)
    
    print("=" * 60)
    print("Visualization script completed!")
    print(f"Output saved to: {args.output_dir}")
    print("=" * 60)
    
    # Output generated chart files
    print("\nGenerated visualization files:")
    
    # Display PNG files
    print("PNG files:")
    if os.path.exists(png_dir):
        for file in os.listdir(png_dir):
            if file.endswith('.png') and 'accuracy' in file:
                print(f" - {file}")
    
    # Display SVG files
    print("\nSVG files:")
    if os.path.exists(svg_dir):
        for file in os.listdir(svg_dir):
            if file.endswith('.svg') and 'accuracy' in file:
                print(f" - {file}")

if __name__ == "__main__":
    main() 