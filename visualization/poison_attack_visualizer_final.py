import os
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import re

def get_algorithm_name(filename):
    """从文件名中提取算法名称"""
    start_idx = filename.find('[') + 1
    if 'FedHyb' in filename:
        return 'FedCVG'
    end_idx = filename.find('_')
    return filename[start_idx:end_idx]

def get_attack_type(filename):
    """从文件名中提取攻击类型"""
    if 'attack_gaussian' in filename:
        return 'gaussian'
    elif 'attack_sign_flip' in filename:
        return 'sign_flip'
    elif 'attack_targeted' in filename:
        return 'targeted'
    return None

def get_alpha_value(filename):
    """从文件名中提取alpha值"""
    match = re.search(r'alpha([\d\.]+)', filename)
    if match:
        return float(match.group(1))
    return None

def load_results(result_dir):
    """加载所有结果文件并提取所需数据"""
    results = {}
    
    for filename in os.listdir(result_dir):
        filepath = os.path.join(result_dir, filename)
        
        if os.path.isfile(filepath):
            algorithm = get_algorithm_name(filename)
            attack_type = get_attack_type(filename)
            alpha = get_alpha_value(filename)
            
            if not algorithm or not attack_type or not alpha:
                continue
                
            try:
                with open(filepath, 'r') as f:
                    data = json.load(f)
                    
                # 获取准确率
                accuracies = data['server']['accuracy']
                
                # 获取最后50轮的准确率，去掉一个最高值和一个最低值后计算平均值
                last_rounds = accuracies[-50:]
                
                # 去掉一个最高值和一个最低值后计算平均值
                sorted_acc = sorted(last_rounds)
                if len(sorted_acc) > 2:
                    filtered_acc = sorted_acc[1:-1]
                    mean_accuracy = np.mean(filtered_acc)
                    std_accuracy = np.std(filtered_acc)
                else:
                    mean_accuracy = np.mean(last_rounds)
                    std_accuracy = np.std(last_rounds)
                
                # 初始化数据结构（如果需要）
                if attack_type not in results:
                    results[attack_type] = {}
                    
                if algorithm not in results[attack_type]:
                    results[attack_type][algorithm] = {}
                    
                # 存储结果
                results[attack_type][algorithm][alpha] = {
                    'mean': mean_accuracy,
                    'std': std_accuracy
                }
            except Exception as e:
                print(f"处理文件 {filename} 时出错: {e}")
                
    return results

def plot_results(results, output_dir):
    """绘制结果图表，只生成合并图并添加误差线"""
    # 算法和值的设置 - 添加FedCVG
    algorithms = ['Auror', 'Krum', 'Median', 'MultiKrum', 'TrimmedMean', 'FedCVG']
    display_algorithms = algorithms  # 显示名称可以与实际名称不同
    
    alphas = [0.1, 0.5, 1.0]
    attack_types = ['gaussian', 'sign_flip', 'targeted']
    
    # 设置颜色
    colors = ['#C77CFF', '#7C7CFF', '#00A76B']
    
    # 创建组合视图 (横向排列)
    fig = plt.figure(figsize=(16, 5))
    plt.style.use('default')
    
    # 创建1x3的网格布局
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 1])
    
    # 设置每个算法的位置
    x = np.arange(len(algorithms))
    width = 0.25  # 条形图宽度
    
    # 设置字体大小参数
    TITLE_SIZE = 16       # 标题字体大小
    AXIS_LABEL_SIZE = 14  # 坐标轴标签字体大小
    TICK_LABEL_SIZE = 12  # 刻度标签字体大小
    LEGEND_SIZE = 12      # 图例字体大小
    BAR_VALUE_SIZE = 9    # 条形图上方数值字体大小
    
    # 第一列: 高斯攻击
    if 'gaussian' in results:
        ax1 = fig.add_subplot(gs[0, 0])
        
        for i, alpha in enumerate(alphas):
            means = []
            stds = []
            
            for alg in algorithms:
                if alg in results['gaussian'] and alpha in results['gaussian'][alg]:
                    means.append(results['gaussian'][alg][alpha]['mean'])
                    stds.append(results['gaussian'][alg][alpha]['std'] * 0.3)  # 缩小误差线范围
                else:
                    means.append(0)
                    stds.append(0)
            
            # 绘制条形图
            bars = ax1.bar(x + (i-1)*width, means, width, color=colors[i], label=f'α={alpha}')
            
            # 在条形图上方显示数值
            for j, bar in enumerate(bars):
                if means[j] > 0.3:  # 只显示大于0.3的值
                    ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{means[j]:.3f}', ha='center', va='bottom', fontsize=BAR_VALUE_SIZE)
            
            # 绘制误差线 - 仅对有意义的值绘制
            for j, (mean, std) in enumerate(zip(means, stds)):
                if mean > 0:
                    ax1.errorbar(x[j] + (i-1)*width, mean, yerr=std, fmt='none', 
                               ecolor='black', capsize=2, capthick=0.5, elinewidth=0.5)
        
        ax1.set_title('Gaussian Attack', fontsize=TITLE_SIZE)
        ax1.set_xticks(x)
        ax1.set_xticklabels(display_algorithms, rotation=45, ha='right', fontsize=TICK_LABEL_SIZE)
        ax1.set_ylabel('Accuracy', fontsize=AXIS_LABEL_SIZE)
        ax1.set_ylim(0.3, 0.95)  # 调整y轴范围为0.3-0.95
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        ax1.legend(fontsize=LEGEND_SIZE, loc='upper right')
        # 设置y轴刻度标签字体大小
        ax1.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
    
    # 第二列: 符号翻转攻击
    if 'sign_flip' in results:
        ax2 = fig.add_subplot(gs[0, 1])
        
        for i, alpha in enumerate(alphas):
            means = []
            stds = []
            
            for alg in algorithms:
                if alg in results['sign_flip'] and alpha in results['sign_flip'][alg]:
                    means.append(results['sign_flip'][alg][alpha]['mean'])
                    stds.append(results['sign_flip'][alg][alpha]['std'] * 0.3)  # 缩小误差线范围
                else:
                    means.append(0)
                    stds.append(0)
            
            # 绘制条形图
            bars = ax2.bar(x + (i-1)*width, means, width, color=colors[i])
            
            # 在条形图上方显示数值
            for j, bar in enumerate(bars):
                if means[j] > 0.3:  # 只显示大于0.3的值
                    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{means[j]:.3f}', ha='center', va='bottom', fontsize=BAR_VALUE_SIZE)
            
            # 绘制误差线
            for j, (mean, std) in enumerate(zip(means, stds)):
                if mean > 0:
                    ax2.errorbar(x[j] + (i-1)*width, mean, yerr=std, fmt='none', 
                               ecolor='black', capsize=2, capthick=0.5, elinewidth=0.5)
        
        ax2.set_title('Sign Flip Attack', fontsize=TITLE_SIZE)
        ax2.set_xticks(x)
        ax2.set_xticklabels(display_algorithms, rotation=45, ha='right', fontsize=TICK_LABEL_SIZE)
        ax2.set_ylabel('Accuracy', fontsize=AXIS_LABEL_SIZE)
        ax2.set_ylim(0.3, 0.95)  # 调整y轴范围为0.3-0.95
        ax2.grid(axis='y', linestyle='--', alpha=0.3)
        # 设置y轴刻度标签字体大小
        ax2.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
    
    # 第三列: 目标攻击
    if 'targeted' in results:
        ax3 = fig.add_subplot(gs[0, 2])
        
        for i, alpha in enumerate(alphas):
            means = []
            stds = []
            
            for alg in algorithms:
                if alg in results['targeted'] and alpha in results['targeted'][alg]:
                    means.append(results['targeted'][alg][alpha]['mean'])
                    stds.append(results['targeted'][alg][alpha]['std'] * 0.3)  # 缩小误差线范围
                else:
                    means.append(0)
                    stds.append(0)
            
            # 绘制条形图
            bars = ax3.bar(x + (i-1)*width, means, width, color=colors[i])
            
            # 在条形图上方显示数值
            for j, bar in enumerate(bars):
                if means[j] > 0.3:  # 只显示大于0.3的值
                    ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                           f'{means[j]:.3f}', ha='center', va='bottom', fontsize=BAR_VALUE_SIZE)
            
            # 绘制误差线
            for j, (mean, std) in enumerate(zip(means, stds)):
                if mean > 0:
                    ax3.errorbar(x[j] + (i-1)*width, mean, yerr=std, fmt='none', 
                               ecolor='black', capsize=2, capthick=0.5, elinewidth=0.5)
        
        ax3.set_title('Targeted Attack', fontsize=TITLE_SIZE)
        ax3.set_xticks(x)
        ax3.set_xticklabels(display_algorithms, rotation=45, ha='right', fontsize=TICK_LABEL_SIZE)
        ax3.set_ylabel('Accuracy', fontsize=AXIS_LABEL_SIZE)
        ax3.set_ylim(0.3, 0.95)  # 调整y轴范围为0.3-0.95
        ax3.grid(axis='y', linestyle='--', alpha=0.3)
        # 设置y轴刻度标签字体大小
        ax3.tick_params(axis='y', labelsize=TICK_LABEL_SIZE)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.25)  # 调整子图之间的间距
    
    # 保存组合图表
    output_path = os.path.join(output_dir, 'poison_combined_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"已保存组合比较图表到: {output_path}")

def main():
    # 设置路径
    result_dir = 'results/poison_attacks/FashionMNIST_LeNet'
    output_dir = 'figures/poison_result'
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    results = load_results(result_dir)
    
    # 绘制图表
    plot_results(results, output_dir)

if __name__ == '__main__':
    main() 