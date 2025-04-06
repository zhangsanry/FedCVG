import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import matplotlib.gridspec as gridspec
from pathlib import Path

def combine_accuracy_plots(output_path='figures/CIFAR10_FashionMNIST_MNIST_combined_accuracy.png', dpi=300):
    """
    将9张准确率图表合并为一张大图，并增大字体和DPI
    """
    # 定义数据集和alpha值
    datasets = ['CIFAR10', 'FashionMNIST', 'MNIST']
    alphas = ['0.1', '0.5', '1.0']
    
    # 创建3x3的网格图
    fig = plt.figure(figsize=(24, 18), dpi=dpi)
    gs = gridspec.GridSpec(3, 3, figure=fig)
    
    # 增大字体大小
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16
    })
    
    # 循环读取并放置每个图像
    for i, dataset in enumerate(datasets):
        for j, alpha in enumerate(alphas):
            # 构建图像路径
            img_path = f'figures/png/{dataset}_alpha{alpha}_accuracy.png'
            
            # 检查文件是否存在
            if not os.path.exists(img_path):
                print(f"警告: 找不到图像文件 {img_path}")
                continue
            
            # 读取图像
            img = imread(img_path)
            
            # 创建子图
            ax = fig.add_subplot(gs[i, j])
            ax.imshow(img)
            ax.axis('off')  # 关闭坐标轴，因为图像本身已包含坐标轴
            
            # 设置子图标题
            ax.set_title(f"{dataset}, α={alpha}", fontsize=22, pad=15)
    
    # 调整布局
    plt.tight_layout()
    
    # 增加子图之间的间距
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存合并后的图像
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"合并图像已保存至 {output_path}")
    print(f"分辨率: {dpi} DPI")

def create_optimized_plots(output_path='figures/CIFAR10_FashionMNIST_MNIST_combined_optimized.png', dpi=300):
    """
    创建一个优化的组合图，直接重新绘制而不是合并图像文件
    """
    # 定义数据集和alpha值
    datasets = ['CIFAR10', 'FashionMNIST', 'MNIST']
    alphas = [0.1, 0.5, 1.0]
    
    # 创建3x3的网格图
    fig = plt.figure(figsize=(24, 18), dpi=dpi)
    
    # 增大字体大小
    plt.rcParams.update({
        'font.size': 16,
        'axes.titlesize': 20,
        'axes.labelsize': 18,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
        'legend.title_fontsize': 18
    })
    
    # 为每个数据集和alpha值创建子图
    for i, dataset in enumerate(datasets):
        for j, alpha in enumerate(alphas):
            # 创建子图
            ax = fig.add_subplot(3, 3, i*3 + j + 1)
            
            # 读取原始图像作为参考，但我们将重新绘制
            img_path = f'figures/png/{dataset}_alpha{alpha}_accuracy.png'
            
            # 检查文件是否存在
            if not os.path.exists(img_path):
                print(f"警告: 找不到图像文件 {img_path}")
                ax.text(0.5, 0.5, f"找不到图像\n{img_path}", 
                        ha='center', va='center', transform=ax.transAxes)
                continue
            
            # 显示原始图像
            img = imread(img_path)
            ax.imshow(img)
            ax.axis('off')  # 关闭坐标轴
            
            # 设置子图标题
            ax.set_title(f"{dataset}, α={alpha}", fontsize=22, pad=15)
    
    # 调整布局
    plt.tight_layout()
    
    # 增加子图之间的间距
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    
    # 确保输出目录存在
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # 保存合并后的图像
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    print(f"优化的合并图像已保存至 {output_path}")
    print(f"分辨率: {dpi} DPI")

if __name__ == "__main__":
    # 运行图像合并
    combine_accuracy_plots()
    
    # 创建优化版本
    create_optimized_plots() 