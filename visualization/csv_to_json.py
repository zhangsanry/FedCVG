import os
import json
import pandas as pd
import argparse
import glob
import re

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='将结果文件转换为CSV格式')
    parser.add_argument('--results_dir', type=str, default='./results',
                        help='包含实验结果的目录')
    parser.add_argument('--output_dir', type=str, default='./figures/csv',
                        help='输出CSV文件的目录')
    parser.add_argument('--datasets', type=str, nargs='+', 
                        default=['MNIST', 'FashionMNIST', 'CIFAR10'],
                        help='要处理的数据集列表')
    return parser.parse_args()

def extract_metadata(file_name):
    """从文件名中提取元数据"""
    # 去掉方括号
    file_name = file_name.replace('[', '').replace(']', '')
    
    # 分割文件名
    parts = file_name.split('_')
    
    # 提取算法名
    algorithm = parts[0]
    
    # 提取alpha值
    alpha = None
    for part in parts:
        if part.startswith('alpha'):
            alpha = float(part[5:])
            break
    
    # 提取数据集名
    dataset = None
    for dataset_name in ['MNIST', 'FashionMNIST', 'CIFAR10', 'Cora', 'Citeseer', 'Pubmed']:
        if dataset_name in file_name:
            dataset = dataset_name
            break
    
    # 提取非IID类型
    non_iid_type = None
    if 'non_iid_label' in file_name:
        non_iid_type = 'label'
    elif 'non_iid_dir' in file_name:
        non_iid_type = 'dir'
    
    return {
        'algorithm': algorithm,
        'dataset': dataset,
        'alpha': alpha,
        'non_iid_type': non_iid_type
    }

def process_result_file(file_path):
    """处理单个结果文件"""
    try:
        # 读取文件内容
        with open(file_path, 'r') as f:
            content = f.read()
        
        # 解析JSON
        data = json.loads(content)
        server_data = data.get('server', {})
        
        # 提取准确率和F1分数
        accuracy = server_data.get('accuracy', [])
        f1_score = server_data.get('f1_score', [])
        train_loss = server_data.get('train_loss', [])
        
        # 创建DataFrame
        df = pd.DataFrame({
            'round': list(range(1, len(accuracy) + 1)),
            'accuracy': accuracy,
            'f1_score': f1_score,
            'train_loss': train_loss
        })
        
        # 提取元数据
        metadata = extract_metadata(os.path.basename(file_path))
        
        # 添加元数据列
        for key, value in metadata.items():
            df[key] = value
        
        return df
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return None

def main():
    """主函数"""
    args = parse_args()
    
    print(f"从 {args.results_dir} 加载实验结果")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 处理每个数据集
    for dataset in args.datasets:
        # 查找数据集目录
        dataset_dir = os.path.join(args.results_dir, f"{dataset}_*")
        dataset_dirs = glob.glob(dataset_dir)
        
        if not dataset_dirs:
            print(f"警告: 未找到 {dataset} 的结果目录")
            continue
        
        all_results = []
        
        # 处理每个数据集目录
        for dir_path in dataset_dirs:
            # 查找所有结果文件
            result_files = glob.glob(os.path.join(dir_path, "*"))
            
            for file_path in result_files:
                df = process_result_file(file_path)
                if df is not None:
                    all_results.append(df)
        
        if all_results:
            # 合并所有结果
            combined_df = pd.concat(all_results, ignore_index=True)
            
            # 保存为CSV
            output_path = os.path.join(args.output_dir, f"{dataset}_results.csv")
            combined_df.to_csv(output_path, index=False)
            print(f"{dataset} 结果已保存至: {output_path}")
    
    # 合并所有数据集的结果
    all_csv_files = glob.glob(os.path.join(args.output_dir, "*_results.csv"))
    if all_csv_files:
        all_data = []
        for csv_file in all_csv_files:
            df = pd.read_csv(csv_file)
            all_data.append(df)
        
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "all_results.csv")
        combined_df.to_csv(combined_path, index=False)
        print(f"所有结果已合并保存至: {combined_path}")
    
    print("\n转换完成。")

if __name__ == "__main__":
    main() 