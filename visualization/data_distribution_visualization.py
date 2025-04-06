import os
import subprocess
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run Data Distribution Visualization')
    parser.add_argument('--dataset', type=str, default='CIFAR10', 
                        choices=['MNIST', 'FashionMNIST', 'CIFAR10'],
                        help='Dataset name (default: CIFAR10)')
    parser.add_argument('--alphas', type=float, nargs='+', default=[0.1, 0.5, 1.0],
                        help='List of alpha values to visualize (default: 0.1 0.5 1.0)')
    parser.add_argument('--num_clients', type=int, default=10,
                        help='Number of clients (default: 10)')
    parser.add_argument('--output_dir', type=str, default='./figures',
                        help='Output directory for figures (default: ./figures)')
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    
    print("=" * 60)
    print(f"Starting Data Distribution Visualization for {args.dataset} dataset...")
    print("=" * 60)
    print(f"Alpha values: {args.alphas}")
    print(f"Number of clients: {args.num_clients}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 60)
    
    # 运行可视化脚本
    for alpha in args.alphas:
        print(f"\n--- Processing alpha={alpha} ---")
        cmd = (f"python visualization/visualize_data_distribution.py --dataset {args.dataset} "
               f"--alpha {alpha} --num_clients {args.num_clients} "
               f"--output_dir {args.output_dir}")
        
        print(f"Executing command: {cmd}")
        subprocess.run(cmd, shell=True)
    
    print(f"\nAll visualizations complete! Results saved in {args.output_dir} directory")
    
    # 显示生成的文件
    print("\nGenerated files:")
    for file in os.listdir(args.output_dir):
        if file.startswith(args.dataset) and (file.endswith('.png') or file.endswith('.svg')):
            print(f" - {file}")

if __name__ == "__main__":
    main() 