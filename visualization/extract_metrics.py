import os
import json
import numpy as np
import pandas as pd
import argparse
import re
from collections import defaultdict

# 常量定义
DATASETS = ['MNIST', 'FashionMNIST', 'CIFAR10']
ALPHAS = [0.1, 0.5, 1.0, 'iid']  # 非IID程度，添加IID

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='提取联邦学习实验的最高指标')
    parser.add_argument('--results_dir', type=str, default='./results',
                       help='结果文件目录')
    parser.add_argument('--output_dir', type=str, default='./figures/csv',
                       help='输出CSV文件目录')
    return parser.parse_args()

def load_result_file(file_path):
    """加载结果文件，返回JSON数据"""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            data = json.loads(content)
        return data
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"警告: 无法加载结果文件 {file_path}，错误：{e}")
        return None

def get_file_list(directory):
    """获取目录中的所有文件"""
    if not os.path.exists(directory):
        return []
    return os.listdir(directory)

def get_available_algorithms(results_dir, dataset, alpha):
    """获取指定数据集和alpha值可用的算法列表"""
    model_name = 'ResNet34' if dataset == 'CIFAR10' else 'LeNet'
    dataset_dir = os.path.join(results_dir, f"{dataset}_{model_name}")
    
    if not os.path.exists(dataset_dir):
        print(f"警告: 目录 {dataset_dir} 不存在")
        return []
    
    algorithms = []
    
    if alpha == 'iid':
        # IID模式
        pattern = re.compile(f"\\[(.*?)_{model_name}_iid_\\d+\\]")
    else:
        # 常规模式 - Non-IID
        pattern = re.compile(f"\\[(.*?)_{model_name}_non_iid_label_alpha{alpha}_\\d+\\]")
        # FedCVG特殊模式(原FedHyb)
        fedhyb_pattern = re.compile(f"\\[(FedHyb_SCAFFOLD_Cluster_Rep_GradMem)_{model_name}_non_iid_label_alpha{alpha}_\\d+\\]")
    
    for file_name in os.listdir(dataset_dir):
        if alpha != 'iid':
            # 尝试匹配FedCVG特殊模式
            match = fedhyb_pattern.match(file_name)
            if match:
                # 将FedHyb改名为FedCVG
                algo = match.group(1).replace("FedHyb", "FedCVG")
                if algo not in algorithms:
                    algorithms.append(algo)
                continue
        
        # 尝试匹配常规模式
        match = pattern.match(file_name)
        if match:
            algo = match.group(1)
            if algo not in algorithms:
                algorithms.append(algo)
    
    return algorithms

def extract_best_metrics(results_dir):
    """提取每个数据集、算法和非IID程度的最高准确率和F1分数"""
    metrics = []
    
    for dataset in DATASETS:
        model_name = 'ResNet34' if dataset == 'CIFAR10' else 'LeNet'
        dataset_dir = os.path.join(results_dir, f"{dataset}_{model_name}")
        
        if not os.path.exists(dataset_dir):
            print(f"跳过数据集 {dataset}，目录不存在: {dataset_dir}")
            continue
        
        print(f"处理数据集: {dataset}")
        
        for alpha in ALPHAS:
            # 获取可用的算法
            algorithms = get_available_algorithms(results_dir, dataset, alpha)
            
            if not algorithms:
                print(f"  跳过 alpha={alpha}，没有可用的算法")
                continue
            
            print(f"  处理 alpha={alpha}, 找到 {len(algorithms)} 个算法: {', '.join(algorithms)}")
            
            for algorithm in algorithms:
                if alpha == 'iid':
                    # IID数据格式
                    result_file = os.path.join(dataset_dir, f"[{algorithm}_{model_name}_iid_1234]")
                elif algorithm.startswith('FedCVG'):
                    # 将FedCVG转回FedHyb以匹配文件名
                    file_algo_name = algorithm.replace('FedCVG', 'FedHyb')
                    result_file = os.path.join(dataset_dir, f"[{file_algo_name}_{model_name}_non_iid_label_alpha{alpha}_1234]")
                else:
                    result_file = os.path.join(dataset_dir, f"[{algorithm}_{model_name}_non_iid_label_alpha{alpha}_1234]")
                
                if not os.path.exists(result_file):
                    print(f"    跳过 {algorithm}，文件不存在: {result_file}")
                    continue
                
                # 加载结果文件
                data = load_result_file(result_file)
                if not data:
                    continue
                
                # 提取准确率和F1分数
                max_accuracy = 0
                max_f1 = 0
                
                # 如果数据包含server字段
                if 'server' in data and isinstance(data['server'], dict):
                    server_data = data['server']
                    
                    # 提取准确率
                    if 'accuracy' in server_data and server_data['accuracy']:
                        accuracy_list = np.array(server_data['accuracy']) * 100
                        max_accuracy = np.max(accuracy_list)
                    
                    # 提取F1分数
                    if 'f1_score' in server_data and server_data['f1_score']:
                        f1_list = np.array(server_data['f1_score']) * 100
                        max_f1 = np.max(f1_list)
                
                # 将指标添加到列表
                metrics.append({
                    'Dataset': dataset,
                    'Algorithm': algorithm,
                    'Alpha': alpha,
                    'Max_Accuracy': max_accuracy,
                    'Max_F1': max_f1
                })
                
                print(f"    {algorithm}: 最高准确率 = {max_accuracy:.2f}%, 最高F1分数 = {max_f1:.2f}%")
    
    return metrics

def save_metrics(metrics, output_dir):
    """将提取的指标保存为CSV文件"""
    if not metrics:
        print("警告: 没有可用的指标数据")
        return None
    
    # 创建DataFrame
    df = pd.DataFrame(metrics)
    
    # 创建算法名称映射函数
    def simplify_algo_name(name):
        if 'FedCVG' in name:
            return 'FedCVG'
        return name
    
    # 创建简化名称的数据框用于显示
    df_display = df.copy()
    df_display['Algorithm'] = df_display['Algorithm'].apply(simplify_algo_name)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存所有指标
    all_metrics_file = os.path.join(output_dir, 'all_metrics.csv')
    df_display.to_csv(all_metrics_file, index=False)
    print(f"所有指标已保存至: {all_metrics_file}")
    
    # 按数据集保存单独的CSV文件
    for dataset in df_display['Dataset'].unique():
        dataset_df = df_display[df_display['Dataset'] == dataset]
        dataset_file = os.path.join(output_dir, f'{dataset}_metrics.csv')
        dataset_df.to_csv(dataset_file, index=False)
        print(f"{dataset} 指标已保存至: {dataset_file}")
    
    # 创建透视表，列为Alpha值，行为算法，值为准确率
    accuracy_pivot = pd.pivot_table(df_display, values='Max_Accuracy', 
                                    index=['Dataset', 'Algorithm'], 
                                    columns=['Alpha'])
    
    accuracy_pivot_file = os.path.join(output_dir, 'accuracy_pivot.csv')
    accuracy_pivot.to_csv(accuracy_pivot_file)
    print(f"准确率透视表已保存至: {accuracy_pivot_file}")
    
    # 创建透视表，列为Alpha值，行为算法，值为F1分数
    f1_pivot = pd.pivot_table(df_display, values='Max_F1', 
                             index=['Dataset', 'Algorithm'], 
                             columns=['Alpha'])
    
    f1_pivot_file = os.path.join(output_dir, 'f1_pivot.csv')
    f1_pivot.to_csv(f1_pivot_file)
    print(f"F1分数透视表已保存至: {f1_pivot_file}")
    
    # 返回原始数据框以便用于后续分析
    return df

def print_summary(df):
    """打印指标的统计摘要"""
    if df is None or df.empty:
        print("没有可用的指标数据进行汇总")
        return
    
    # 创建算法名称映射函数
    def simplify_algo_name(name):
        if 'FedCVG' in name:
            return 'FedCVG'
        return name
    
    print("\n===== 指标摘要 =====")
    
    # 按数据集和Alpha值分组，找出每组中准确率最高的算法
    group_cols = ['Dataset', 'Alpha']
    best_accuracy = df.loc[df.groupby(group_cols)['Max_Accuracy'].idxmax()]
    
    print("\n最高准确率算法:")
    for _, row in best_accuracy.iterrows():
        algo_display = simplify_algo_name(row['Algorithm'])
        print(f"{row['Dataset']}, Alpha={row['Alpha']}: {algo_display} ({row['Max_Accuracy']:.2f}%)")
    
    # 按数据集和Alpha值分组，找出每组中F1分数最高的算法
    best_f1 = df.loc[df.groupby(group_cols)['Max_F1'].idxmax()]
    
    print("\n最高F1分数算法:")
    for _, row in best_f1.iterrows():
        algo_display = simplify_algo_name(row['Algorithm'])
        print(f"{row['Dataset']}, Alpha={row['Alpha']}: {algo_display} ({row['Max_F1']:.2f}%)")
    
    # 算法在各个数据集和Alpha值下的平均表现
    algo_avg = df.groupby('Algorithm')[['Max_Accuracy', 'Max_F1']].mean().sort_values('Max_Accuracy', ascending=False)
    
    print("\n算法平均表现 (按准确率排序):")
    for algo, row in algo_avg.iterrows():
        algo_display = simplify_algo_name(algo)
        print(f"{algo_display}: 准确率={row['Max_Accuracy']:.2f}%, F1分数={row['Max_F1']:.2f}%")

def main():
    """主函数"""
    args = parse_args()
    
    print(f"从 {args.results_dir} 提取指标")
    print(f"输出将保存至 {args.output_dir}")
    
    # 提取指标
    metrics = extract_best_metrics(args.results_dir)
    
    # 保存指标
    df = save_metrics(metrics, args.output_dir)
    
    # 打印摘要
    print_summary(df)
    
    print("\n指标提取完成!")

if __name__ == "__main__":
    main() 