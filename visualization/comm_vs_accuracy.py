import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.font_manager import FontProperties

# 设置中文字体，避免警告
try:
    # 尝试使用微软雅黑字体
    font = FontProperties(fname=r'C:\Windows\Fonts\msyh.ttc', size=18)  # 进一步增大字体大小
except:
    # 如果找不到，使用系统默认字体
    font = FontProperties(size=18)  # 进一步增大字体大小

# 设置全局字体大小 - 保持标题和坐标轴标签的大尺寸，还原图例大小
TITLE_SIZE = 24       # 标题字体大小
AXIS_LABEL_SIZE = 22  # 坐标轴标签字体大小
TICK_LABEL_SIZE = 20  # 刻度标签字体大小
LEGEND_SIZE = 12      # 图例字体大小 - 还原到原始大小
LINE_WIDTH = 3        # 线条宽度

def read_json_file(file_path):
    """读取JSON文件"""
    with open(file_path, 'r') as f:
        return json.load(f)

def get_max_comm_costs(dataset='CIFAR10', alpha=0.5):
    """获取所有算法中的最大通信开销，用于统一纵轴范围"""
    algorithms = ['Auror', 'FedAvg', 'FedHyb']
    max_costs = []
    
    for algo in algorithms:
        comm_file = f'results/consumption/{algo}_comm_{dataset}_ResNet34_10clients.json'
        try:
            comm_data = read_json_file(comm_file)
            max_cost = max(comm_data['communication_costs']['cumulative'])
            max_costs.append(max_cost)
        except (FileNotFoundError, KeyError) as e:
            print(f"警告: 无法读取{algo}的通信开销数据 - {e}")
    
    # 返回最大值，如果没有有效数据则返回None
    return max(max_costs) if max_costs else None

def plot_comm_vs_accuracy(algo, dataset='CIFAR10', alpha=0.5, output_dir='figures/comm_vs_accuracy', 
                         ax=None, save_individual=True, y2_max=None):
    """
    创建通信开销和准确率对比图
    
    参数:
    - algo: 算法名称 (如 'Auror', 'FedAvg', 'FedHyb')
    - dataset: 数据集名称
    - alpha: 分布参数
    - output_dir: 输出目录
    - ax: 可选的轴对象，用于组合图
    - save_individual: 是否保存单独的图表
    - y2_max: 右侧Y轴（通信开销）的最大值
    """
    # 构建文件路径
    comm_file = f'results/consumption/{algo}_comm_{dataset}_ResNet34_10clients.json'
    
    # 为FedHyb使用特殊路径
    if algo == 'FedHyb':
        accuracy_file = f'results/CIFAR10_ResNet34/[FedHyb_SCAFFOLD_Cluster_Rep_GradMem_ResNet34_non_iid_label_alpha{alpha}_1234]'
    else:
        accuracy_file = f'results/CIFAR10_ResNet34/[{algo}_ResNet34_non_iid_label_alpha{alpha}_1234]'
    
    # 读取通信开销和准确率数据
    try:
        comm_data = read_json_file(comm_file)
        with open(accuracy_file, 'r') as f:
            accuracy_data = json.load(f)
    except FileNotFoundError as e:
        print(f"错误: 无法找到文件 - {e}")
        return
    
    # 获取通信开销数据
    rounds = list(range(1, len(comm_data['communication_costs']['cumulative']) + 1))
    comm_costs = comm_data['communication_costs']['cumulative']
    
    # 获取准确率数据
    accuracies = accuracy_data['server']['accuracy']
    
    # 截断数据使两者长度一致
    min_length = min(len(rounds), len(accuracies))
    rounds = rounds[:min_length]
    comm_costs = comm_costs[:min_length]
    accuracies = accuracies[:min_length]
    
    # 如果没有提供ax，创建新的图表
    if ax is None:
        fig = plt.figure(figsize=(12, 8))  # 增大图表大小
        ax1 = fig.add_subplot(111)
    else:
        ax1 = ax
        
    # 第二个Y轴用于通信开销
    ax2 = ax1.twinx()
    
    # 绘制准确率曲线 - 用蓝色，增大线宽
    ln1 = ax1.plot(rounds, [acc * 100 for acc in accuracies], 'b-', linewidth=LINE_WIDTH, label='Accuracy (%)')
    
    # 绘制通信开销曲线 - 用橙色，增大线宽和标记大小
    ln2 = ax2.plot(rounds, comm_costs, 'o-', color='#FF9900', linewidth=LINE_WIDTH, markersize=8, label='Communication Cost (MB)')
    
    # 填充通信开销区域
    ax2.fill_between(rounds, 0, comm_costs, alpha=0.2, color='#FF9900')  # 稍微增加填充透明度
    
    # 标题和标签 - 增大字体大小
    # 如果是FedHyb，显示为FedCVG
    display_name = 'FedCVG' if algo == 'FedHyb' else algo
    ax1.set_title(f'{display_name} on {dataset} (α={alpha})', fontsize=TITLE_SIZE, pad=20)  # 增加标题和图表之间的距离
    ax1.set_xlabel('Communication Round', fontsize=AXIS_LABEL_SIZE, labelpad=15)  # 增加标签和轴之间的距离
    ax1.set_ylabel('Accuracy (%)', fontsize=AXIS_LABEL_SIZE, labelpad=15)  # 增加标签和轴之间的距离
    ax2.set_ylabel('Cumulative Communication Cost (MB)', fontsize=AXIS_LABEL_SIZE, labelpad=15)  # 增加标签和轴之间的距离
    
    # 设置刻度标签字体大小
    ax1.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE, length=6, width=2, pad=10)  # 增大刻度大小和间距
    ax2.tick_params(axis='both', which='major', labelsize=TICK_LABEL_SIZE, length=6, width=2, pad=10)  # 增大刻度大小和间距
    
    # 设置网格线
    ax1.grid(True, linestyle='--', alpha=0.7, linewidth=1.5)  # 增加网格线宽度
    
    # 设置Y轴范围
    if dataset == 'CIFAR10':
        if alpha == 0.1:
            ax1.set_ylim(10, 60)
        elif alpha == 0.5:
            ax1.set_ylim(50, 80)
        else:  # alpha == 1.0
            ax1.set_ylim(70, 85)
    
    # 统一右侧Y轴的最大值
    if y2_max is not None:
        ax2.set_ylim(0, y2_max * 1.05)  # 给顶部留一些空间
    
    # 合并图例 - 还原为原始样式
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc='upper left', fontsize=LEGEND_SIZE)
    
    # 如果没有提供ax且需要保存单独的图表，则保存
    if ax is None and save_individual:
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f'{algo}_{dataset}_alpha{alpha}_comm_vs_accuracy.png')
        plt.savefig(output_path, dpi=400, bbox_inches='tight')  # 增加DPI
        print(f"图表已保存至: {output_path}")
        plt.close()
    
    return ax1, ax2

def create_combined_row_plot(dataset='CIFAR10', alpha=0.5, output_dir='figures/comm_vs_accuracy'):
    """
    创建一行三列的组合图表，保持原有长宽比，统一纵轴
    
    参数:
    - dataset: 数据集名称
    - alpha: 分布参数
    - output_dir: 输出目录
    """
    # 要比较的算法
    algorithms = ['Auror', 'FedAvg', 'FedHyb']
    
    # 获取所有算法中最大的通信开销值，用于统一Y轴
    max_comm_cost = get_max_comm_costs(dataset, alpha)
    if max_comm_cost is None:
        print("警告: 无法确定通信开销的最大值，将使用各图表的独立范围")
    
    # 创建大图
    fig = plt.figure(figsize=(36, 8))  # 增大图表尺寸
    
    # 创建网格
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # 为每个算法创建子图
    for i, algo in enumerate(algorithms):
        ax = fig.add_subplot(gs[0, i])
        plot_comm_vs_accuracy(algo, dataset, alpha, output_dir, ax=ax, save_individual=False, y2_max=max_comm_cost)
    
    # 调整布局，增加子图之间的间距
    plt.tight_layout(pad=4.0)
    
    # 保存组合图表
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f'{dataset}_alpha{alpha}_combined_row.png')
    plt.savefig(output_path, dpi=400, bbox_inches='tight')  # 增加DPI
    print(f"行式组合图表已保存至: {output_path}")
    plt.close()

def main():
    """主函数"""
    # 确保输出目录存在
    output_dir = 'figures/comm_vs_accuracy'
    os.makedirs(output_dir, exist_ok=True)
    
    # CIFAR10数据集上的不同算法
    dataset = 'CIFAR10'
    alpha = 0.5  # 使用alpha=0.5的结果
    algorithms = ['Auror', 'FedAvg', 'FedHyb']
    
    # 获取最大通信开销值，用于统一Y轴
    max_comm_cost = get_max_comm_costs(dataset, alpha)
    
    # 为每个算法创建单独的图表
    for algo in algorithms:
        plot_comm_vs_accuracy(algo, dataset, alpha, output_dir, y2_max=max_comm_cost)
    
    # 创建一行三列的组合图表
    create_combined_row_plot(dataset, alpha, output_dir)

if __name__ == "__main__":
    main() 