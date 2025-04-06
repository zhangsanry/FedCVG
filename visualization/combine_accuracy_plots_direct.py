import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
import re
import sys

# 定义颜色映射，为算法分配固定颜色，保持可视化一致性
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
    'FedCVG_SCAFFOLD_Cluster_Rep_GradMem': '#FF1493',  # 亮粉色
    'FedCVG_FedProx': '#00FFFF'    # 青色
}

# 要比较的主要算法（按照图例顺序显示，仅用于排序）
MAIN_ALGORITHMS = ['FedAvg', 'Krum', 'SCAFFOLD', 'Auror', 'FedProx', 'Median', 'TrimmedMean', 'Bulyan', 'MultiKrum', 'FedCVG_SCAFFOLD_Cluster_Rep_GradMem', 'FedCVG_SCAFFOLD', 'FedCVG_FedProx']

def load_result_file(file_path):
    """从结果文件中读取数据"""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
            data = json.loads(content)
            return data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"加载错误 {file_path}: {e}")
        return None

def get_available_algorithms(results_dir, dataset, alpha):
    """获取指定数据集和alpha值下可用的算法列表"""
    dataset_dir = os.path.join(results_dir, dataset)
    if not os.path.exists(dataset_dir):
        print(f"数据集目录不存在: {dataset_dir}")
        return []
    
    algorithms = []
    
    if alpha == 'iid':
        # IID模式
        pattern = rf"\[(.*?)_{dataset.split('_')[1]}_iid_1234\]"
        
        for file_name in os.listdir(dataset_dir):
            # 尝试匹配IID模式
            match = re.match(pattern, file_name)
            if match:
                algo_name = match.group(1)
                algorithms.append(algo_name)
    else:
        # 常规算法的模式
        pattern = rf"\[(.*?)_{dataset.split('_')[1]}_non_iid_label_alpha{alpha}_1234\]"
        # FedCVG特殊模式 (原FedHyb)
        fedhyb_pattern = rf"\[(FedHyb.*?)_{dataset.split('_')[1]}_non_iid_label_alpha{alpha}_1234\]"
        
        for file_name in os.listdir(dataset_dir):
            # 尝试匹配FedCVG特殊模式
            match = re.match(fedhyb_pattern, file_name)
            if match:
                # 将FedHyb改名为FedCVG
                algo_name = match.group(1).replace("FedHyb", "FedCVG")
                algorithms.append(algo_name)
                continue
                
            # 尝试匹配常规模式
            match = re.match(pattern, file_name)
            if match:
                algo_name = match.group(1)
                algorithms.append(algo_name)
    
    return algorithms

def draw_accuracy_subplot(ax, dataset, alpha, results_dir, select_algorithms=None, max_algorithms=0):
    """为特定数据集和alpha值绘制精确度曲线子图"""
    # 设置图表样式
    sns.set_style("whitegrid")
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 获取当前数据集和alpha值下可用的算法
    available_algorithms = get_available_algorithms(results_dir, dataset, alpha)
    
    if not available_algorithms:
        print(f"没有可用的算法: {dataset}, α={alpha}")
        ax.text(0.5, 0.5, f"没有找到数据", ha='center', va='center', transform=ax.transAxes)
        return False
    
    # 如果指定了要显示的算法，仅保留这些算法
    if select_algorithms:
        algorithms_to_show = [algo for algo in available_algorithms if algo in select_algorithms]
        if not algorithms_to_show:
            print(f"指定的算法 {select_algorithms} 中没有一个可用于 {dataset}, α={alpha}")
            print(f"可用算法: {available_algorithms}")
            algorithms_to_show = available_algorithms
    else:
        # 默认显示所有可用算法
        algorithms_to_show = available_algorithms
    
    # 如果指定了最大显示算法数，限制显示的算法数量
    if max_algorithms > 0 and len(algorithms_to_show) > max_algorithms:
        # 优先选择主要算法
        main_algos = [algo for algo in MAIN_ALGORITHMS if algo in algorithms_to_show][:max_algorithms]
        if len(main_algos) < max_algorithms:
            # 添加其他算法直到达到最大数量
            other_algos = [algo for algo in algorithms_to_show if algo not in main_algos]
            algorithms_to_show = main_algos + other_algos[:max_algorithms - len(main_algos)]
        else:
            algorithms_to_show = main_algos[:max_algorithms]
    
    max_rounds = 0
    valid_results = False
    dataset_dir = os.path.join(results_dir, dataset)
    
    # 排序算法以保持一致的图例顺序
    def sort_key(algo):
        try:
            return MAIN_ALGORITHMS.index(algo)
        except ValueError:
            # 对于不在主要算法列表中的算法，如果是以FedCVG开头的，给予较高优先级
            if algo.startswith('FedCVG'):
                return len(MAIN_ALGORITHMS) - 0.5  # 给FedCVG算法较高的优先级
            # 其他算法按照在algorithms_to_show中的顺序排列
            return len(MAIN_ALGORITHMS) + 1
    
    algorithms_to_show.sort(key=sort_key)
    
    # 为每个算法分配一种线型，以便在算法数量较多时更容易区分
    linestyles = ['-', '--', '-.', ':']
    
    # 标记已经添加到图例中的FedCVG
    fedcvg_added = False
    
    for i, algo_name in enumerate(algorithms_to_show):
        # 构建文件路径
        if alpha == 'iid':
            # IID数据格式
            result_file = f"[{algo_name}_{dataset.split('_')[1]}_iid_1234]"
        elif 'FedCVG' in algo_name:
            # 将FedCVG转换回原始文件名FedHyb
            result_file = f"[{algo_name.replace('FedCVG', 'FedHyb')}_{dataset.split('_')[1]}_non_iid_label_alpha{alpha}_1234]"
        else:
            result_file = f"[{algo_name}_{dataset.split('_')[1]}_non_iid_label_alpha{alpha}_1234]"
        result_path = os.path.join(dataset_dir, result_file)
        
        print(f"尝试读取: {result_path}")
        
        # 读取结果数据
        data = load_result_file(result_path)
        if data and 'server' in data and 'accuracy' in data['server']:
            accuracy = data['server']['accuracy']
            
            # 将准确率值乘以100转换为百分比
            accuracy = [acc * 100 for acc in accuracy]
            
            rounds = list(range(len(accuracy)))
            max_rounds = max(max_rounds, len(rounds))
            
            # 获取算法颜色，如果未定义则使用随机颜色
            color = COLOR_MAP.get(algo_name, None)
            if color is None:
                # 为未定义的算法分配颜色
                color_idx = i % len(sns.color_palette("husl", len(algorithms_to_show)))
                color = sns.color_palette("husl", len(algorithms_to_show))[color_idx]
            
            # 选择线型，为FedCVG算法使用实线
            if 'FedCVG' in algo_name:
                linestyle = '-'
                linewidth = 3  # 使用粗线宽以突出显示
                # 简化FedCVG显示名称，并避免重复
                if fedcvg_added:
                    # 如果已经添加过FedCVG，不再显示标签
                    display_name = '_nolegend_'
                else:
                    display_name = 'FedCVG'
                    fedcvg_added = True
            else:
                linestyle = linestyles[i % len(linestyles)]
                linewidth = 2
                display_name = algo_name
            
            # 绘制精确度曲线
            ax.plot(rounds, accuracy, 
                    label=display_name, 
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth)
            valid_results = True
    
    if not valid_results:
        print(f"未找到有效结果: {dataset}, α={alpha}")
        ax.text(0.5, 0.5, f"未找到有效结果", ha='center', va='center', transform=ax.transAxes)
        return False
    
    # 设置标签
    ax.set_xlabel('Communication Round', fontsize=18)
    ax.set_ylabel('Accuracy (%)', fontsize=18)
    
    # 设置纵坐标范围，根据数据集和alpha值显示不同范围
    if dataset.startswith('CIFAR10'):
        if alpha == '0.1':
            # CIFAR10, α=0.1时显示20-60%
            ax.set_ylim(20, 60)
        elif alpha == '0.5':
            # CIFAR10, α=0.5时显示55-80%
            ax.set_ylim(55, 80)
        else:  # alpha == '1.0'
            # CIFAR10, α=1.0时显示70-85%
            ax.set_ylim(70, 85)
    elif dataset.startswith('FashionMNIST'):
        if alpha == '0.1':
            # FashionMNIST, α=0.1时显示60-90%
            ax.set_ylim(60, 90)
        elif alpha == '0.5':
            # FashionMNIST, α=0.5时显示75-90%
            ax.set_ylim(75, 90)
        else:  # alpha == '1.0'
            # FashionMNIST, α=1.0时显示80-90%
            ax.set_ylim(80, 90)
    else:  # MNIST
        if alpha == '1.0':
            # MNIST, α=1.0时显示97.5-100%
            ax.set_ylim(97.5, 100)
        elif alpha == '0.5':
            # MNIST, α=0.5时显示95-100%
            ax.set_ylim(95, 100)
        else:  # alpha == '0.1'
            # MNIST, α=0.1时显示80-100%
            ax.set_ylim(80, 100)
    
    # 设置横坐标从0到100，间隔10
    max_x = min(100, max_rounds if max_rounds > 0 else 100)
    ax.set_xlim(0, max_x)
    ax.set_xticks(np.arange(0, max_x+1, 10))
    
    # 添加图例
    ax.legend(loc='upper right', frameon=True, ncol=2, fontsize=14)
    
    return True

def create_combined_plot(results_dir='../results', output_path='../figures/CIFAR10_FashionMNIST_MNIST_combined_direct.png', dpi=300):
    """创建直接合并的准确率图表，不经过中间图像文件"""
    print(f"开始创建合并图表...")
    print(f"结果目录: {results_dir}")
    print(f"输出路径: {output_path}")
    
    # 检查结果目录是否存在
    if not os.path.exists(results_dir):
        print(f"错误: 结果目录不存在 - {results_dir}")
        print(f"当前工作目录: {os.getcwd()}")
        return
    
    # 定义数据集和alpha值
    datasets = ['CIFAR10_ResNet34', 'FashionMNIST_LeNet', 'MNIST_LeNet']
    alphas = ['0.1', '0.5', '1.0']
    
    # 增大字体大小
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 22,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 14
    })
    
    # 创建3x3的网格图
    fig = plt.figure(figsize=(24, 18), dpi=dpi)
    
    # 建立子图网格
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # 循环遍历数据集和alpha值，绘制每个子图
    for i, dataset in enumerate(datasets):
        for j, alpha in enumerate(alphas):
            print(f"绘制子图: {dataset}, α={alpha}")
            # 创建子图
            ax = fig.add_subplot(gs[i, j])
            
            # 绘制特定数据集和alpha值的准确率曲线
            draw_accuracy_subplot(ax, dataset, alpha, results_dir)
    
    # 调整布局
    plt.tight_layout()
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存合并后的图像
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"直接合并的准确率图表已保存至 {output_path}")
    print(f"分辨率: {dpi} DPI")

if __name__ == "__main__":
    # 输出当前工作目录和Python路径
    print(f"当前工作目录: {os.getcwd()}")
    print(f"Python路径: {sys.path}")
    
    # 运行直接从源数据绘制的合并图表
    create_combined_plot(dpi=600)  # 使用高DPI以确保图像清晰 