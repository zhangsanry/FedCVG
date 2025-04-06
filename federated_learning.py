import os
import random
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from json import JSONEncoder
from torch.utils.data import DataLoader
from federated_parser import get_args
from preprocessing.baselines_dataloader import load_data, divide_data
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}

class FederatedLearning:
    def __init__(self, args):
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # 设置随机种子
        np.random.seed(args.i_seed)
        torch.manual_seed(args.i_seed)
        random.seed(args.i_seed)
        
        # 加载数据
        self.trainset_config, self.testset = divide_data(
            num_client=args.num_client,
            dataset_name=args.dataset,
            i_seed=args.i_seed,
            distribution_type=args.distribution_type,
            alpha=args.alpha
        )
        
        # 初始化客户端和服务器
        self.init_clients()
        self.init_server()
        
        # 记录训练结果
        self.results = {
            'server': {
                'accuracy': [],
                'train_loss': [],
                'f1_score': []
            }
        }
        
        # 为SCAFFOLD算法初始化控制变量
        if self.args.fed_algo == "SCAFFOLD":
            self.control_variate = {}
            for name, param in self.server_model.state_dict().items():
                self.control_variate[name] = torch.zeros_like(param)
            
            # 为每个客户端初始化控制变量
            for client_id in self.clients:
                self.clients[client_id]['control_variate'] = {}
                for name, param in self.server_model.state_dict().items():
                    self.clients[client_id]['control_variate'][name] = torch.zeros_like(param)
    
    def init_clients(self):
        """初始化所有客户端"""
        self.clients = {}
        for client_id in self.trainset_config['users']:
            self.clients[client_id] = {
                'model': self.create_model(),
                'optimizer': None,
                'data': self.trainset_config['user_data'][client_id]
            }
    
    def init_server(self):
        """初始化服务器"""
        self.server_model = self.create_model()
        self.test_loader = DataLoader(self.testset, batch_size=self.args.batch_size)
    
    def create_model(self):
        """创建模型实例"""
        from models import LeNet, CNN, ResNet18, ResNet34, ResNet50, AlexCifarNet
        
        if self.args.model == 'LeNet':
            return LeNet().to(self.device)
        elif self.args.model == 'CNN':
            return CNN().to(self.device)
        elif self.args.model == 'ResNet18':
            return ResNet18().to(self.device)
        elif self.args.model == 'ResNet34':
            return ResNet34().to(self.device)
        elif self.args.model == 'ResNet50':
            return ResNet50().to(self.device)
        elif self.args.model == 'AlexCifarNet':
            return AlexCifarNet().to(self.device)
        else:
            raise NotImplementedError(f"模型 {self.args.model} 尚未实现")
    
    def client_update(self, client_id):
        """客户端本地训练"""
        client = self.clients[client_id]
        model = client['model']
        model.train()
        
        # 设置优化器
        optimizer = torch.optim.SGD(
            model.parameters(), 
            lr=self.args.learning_rate,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        criterion = torch.nn.CrossEntropyLoss()
        
        # 加载数据
        train_loader = DataLoader(client['data'], batch_size=self.args.batch_size, shuffle=True)
        
        # 如果使用FedProx，保存全局模型参数用于计算近端项
        if self.args.fed_algo == "FedProx":
            global_params = {}
            for name, param in self.server_model.state_dict().items():
                global_params[name] = param.clone().detach()
            
            mu = self.args.fedprox_mu  # 近端项系数
        
        # 如果使用SCAFFOLD，准备控制变量
        if self.args.fed_algo == "SCAFFOLD":
            # 获取当前客户端和服务器的控制变量
            client_control = client['control_variate']
            server_control = self.control_variate
            
            # 保存本地控制变量的副本用于更新
            old_client_control = {k: v.clone() for k, v in client_control.items()}
            
            # 保存参数初始值用于更新控制变量
            init_params = {k: v.clone() for k, v in model.state_dict().items()}
        
        # 本地训练
        epoch_loss = 0
        for epoch in range(self.args.num_local_epoch):
            batch_loss = 0
            batch_count = 0
            
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                optimizer.zero_grad()
                output = model(data)
                
                # 基础损失
                loss = criterion(output, target)
                
                # FedProx: 添加近端项
                if self.args.fed_algo == "FedProx":
                    proximal_term = 0.0
                    for name, param in model.named_parameters():
                        if name in global_params:
                            proximal_term += torch.sum((param - global_params[name])**2)
                    
                    loss += (mu / 2) * proximal_term
                
                loss.backward()
                
                # SCAFFOLD: 应用控制变量修正梯度
                if self.args.fed_algo == "SCAFFOLD":
                    for name, param in model.named_parameters():
                        if param.grad is not None and name in client_control and name in server_control:
                            param.grad += server_control[name] - client_control[name]
                
                optimizer.step()
                batch_loss += loss.item()
                batch_count += 1
                
            epoch_loss = batch_loss / len(train_loader)
        
        # SCAFFOLD: 更新客户端控制变量
        if self.args.fed_algo == "SCAFFOLD":
            for name, param in model.state_dict().items():
                if name in client_control and name in init_params:
                    # 计算新的控制变量: c_i^{+} = c_i - c + (x_0 - x_K) / (K * eta)
                    lr = self.args.learning_rate
                    step_size = self.args.scaffold_c_lr / (self.args.num_local_epoch * lr)
                    client_control[name] = old_client_control[name] - server_control[name] + (init_params[name] - param) * step_size
        
        model_state = model.state_dict()
        
        # 如果启用了投毒攻击且当前客户端是恶意客户端，执行梯度投毒攻击
        if hasattr(self.args, 'enable_attack') and self.args.enable_attack:
            # 从客户端ID中提取客户端编号
            client_idx = -1
            if isinstance(client_id, str) and client_id.startswith('client_'):
                try:
                    client_idx = int(client_id.split('_')[1])
                except (IndexError, ValueError):
                    # 尝试不同的格式
                    try:
                        client_idx = int(''.join(filter(str.isdigit, client_id)))
                    except ValueError:
                        print(f"无法从客户端ID {client_id} 提取编号")
            
            # 检查客户端是否是恶意客户端
            if client_idx != -1 and client_idx < self.args.num_malicious:
                # 获取模型参数差异作为梯度
                client_gradients = {}
                for name, param in model_state.items():
                    if name in self.server_model.state_dict():
                        server_param = self.server_model.state_dict()[name]
                        # 计算梯度（模型差异）
                        gradient = server_param - param
                        
                        # 根据攻击类型修改梯度
                        if self.args.attack_type == 'gaussian':
                            # 添加高斯噪声
                            noise = torch.randn_like(gradient) * self.args.noise_level
                            gradient = gradient + noise
                        elif self.args.attack_type == 'sign_flip':
                            # 翻转梯度符号
                            gradient = -gradient * self.args.noise_level
                        elif self.args.attack_type == 'targeted':
                            # 目标攻击：将梯度朝固定方向偏移
                            gradient = torch.ones_like(gradient) * self.args.noise_level
                        
                        # 更新模型状态
                        model_state[name] = server_param - gradient
                
                print(f"客户端 {client_id} (索引 {client_idx}) 执行了 {self.args.attack_type} 类型的梯度投毒攻击")
        
        return model_state, len(client['data']), epoch_loss
    
    def server_aggregate(self, updates):
        """服务器聚合模型"""
        total_weight = 0
        aggregated_weights = None
        
        # 根据不同的联邦学习算法执行不同的聚合策略
        if self.args.fed_algo == "FedAvg" or self.args.fed_algo == "FedProx":
            # FedAvg/FedProx: 标准的加权平均
            for client_id, (weights, num_samples, _) in updates.items():
                total_weight += num_samples
                if aggregated_weights is None:
                    aggregated_weights = {k: v.clone() * num_samples for k, v in weights.items()}
                else:
                    for k in weights.keys():
                        aggregated_weights[k] += weights[k] * num_samples
            
            # 计算加权平均
            for k in aggregated_weights.keys():
                aggregated_weights[k] = aggregated_weights[k] / total_weight
                
        elif self.args.fed_algo == "SCAFFOLD":
            # SCAFFOLD: 标准的模型平均 + 控制变量更新
            client_control_sum = {}
            
            for client_id, (weights, num_samples, _) in updates.items():
                total_weight += num_samples
                # 聚合模型权重
                if aggregated_weights is None:
                    aggregated_weights = {k: v.clone() * num_samples for k, v in weights.items()}
                else:
                    for k in weights.keys():
                        aggregated_weights[k] += weights[k] * num_samples
                
                # 收集客户端控制变量用于更新服务器控制变量
                client_control = self.clients[client_id]['control_variate']
                if not client_control_sum:
                    client_control_sum = {k: v.clone() * num_samples for k, v in client_control.items()}
                else:
                    for k in client_control.keys():
                        client_control_sum[k] += client_control[k] * num_samples
            
            # 计算模型的加权平均
            for k in aggregated_weights.keys():
                aggregated_weights[k] = aggregated_weights[k] / total_weight
                
            # 更新服务器控制变量: c = (1/n) * sum(c_i)
            for k in client_control_sum.keys():
                self.control_variate[k] = client_control_sum[k] / total_weight
                
        elif self.args.fed_algo == "FedNova":
            # FedNova: 规范化权重平均
            tau_eff_sum = 0
            for client_id, (weights, num_samples, _) in updates.items():
                # 计算有效本地更新次数
                if self.args.fednova_tau_eff == 'uniform':
                    tau_eff = 1.0  # 统一权重
                else:  # 'n_local_epoch'
                    tau_eff = self.args.num_local_epoch
                
                tau_eff_sum += tau_eff
                
                # 初始化聚合权重
                if aggregated_weights is None:
                    aggregated_weights = {k: v.clone() * tau_eff for k, v in weights.items()}
                else:
                    for k in weights.keys():
                        aggregated_weights[k] += weights[k] * tau_eff
            
            # 计算规范化权重平均
            if tau_eff_sum > 0:
                for k in aggregated_weights.keys():
                    aggregated_weights[k] = aggregated_weights[k] / tau_eff_sum
        
        elif self.args.fed_algo == "Krum" or self.args.fed_algo == "MultiKrum":
            # Krum/MultiKrum: 基于Byzantine容错的聚合
            aggregated_weights = self.krum_aggregate(updates)
            
        elif self.args.fed_algo == "Bulyan":
            # Bulyan: 结合Krum距离筛选和逐坐标修剪平均
            aggregated_weights = self.bulyan_aggregate(updates)
            
        elif self.args.fed_algo == "TrimmedMean":
            # TrimmedMean: 逐坐标剪裁平均，每个坐标去掉最大和最小的一些值后平均
            aggregated_weights = self.trimmed_mean_aggregate(updates)
            
        elif self.args.fed_algo == "Median":
            # Median: 逐坐标中位数，特殊的TrimmedMean
            aggregated_weights = self.median_aggregate(updates)
        
        elif self.args.fed_algo == "Auror":
            # Auror: 基于聚类的异常检测，识别并丢弃可能的恶意更新
            aggregated_weights = self.auror_aggregate(updates)
        
        return aggregated_weights
    
    def krum_aggregate(self, updates):
        """Krum和MultiKrum聚合算法实现"""
        # 提取所有客户端的更新
        client_weights = {}
        for client_id, (weights, num_samples, _) in updates.items():
            client_weights[client_id] = weights
        
        # 转换模型参数为向量表示，用于计算距离
        weight_vectors = {}
        for client_id, weights in client_weights.items():
            vector = []
            for name, param in weights.items():
                vector.append(param.view(-1))
            weight_vectors[client_id] = torch.cat(vector)
        
        # 计算客户端模型两两之间的欧氏距离
        num_clients = len(weight_vectors)
        distances = torch.zeros(num_clients, num_clients)
        client_ids = list(weight_vectors.keys())
        
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                client_i = client_ids[i]
                client_j = client_ids[j]
                dist = torch.norm(weight_vectors[client_i] - weight_vectors[client_j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # 计算krum得分：每个客户端与其最近的 n-f-2 个客户端的距离总和
        n = num_clients
        f = min(self.args.num_malicious_tolerance, n-2)  # 最多容忍 f 个拜占庭错误
        m = n - f - 2  # 选择最近的 m 个客户端
        
        if m <= 0:
            print(f"警告: 客户端数量不足，无法应用Krum算法。至少需要 {f+3} 个客户端，但只有 {n} 个。回退到FedAvg。")
            # 回退到FedAvg
            return self.fedavg_aggregate(updates)
        
        scores = torch.zeros(num_clients)
        
        for i in range(num_clients):
            # 获取与当前客户端距离最近的 m 个客户端的距离
            client_distances = distances[i].clone()
            client_distances[i] = float('inf')  # 排除自己
            closest_m_distances, _ = torch.topk(client_distances, m, largest=False)
            # Krum得分是与最近的 m 个客户端的距离总和
            scores[i] = torch.sum(closest_m_distances)
        
        if self.args.fed_algo == "Krum":
            # 原始Krum：选取得分最低（距离和最小）的客户端更新
            if self.args.use_distances_as_weights:
                # 使用距离的倒数作为权重进行加权聚合
                weights = 1.0 / (scores + 1e-10)  # 添加小值避免除零
                weights = weights / weights.sum()  # 归一化
                
                # 加权聚合
                aggregated_weights = {}
                for idx, client_id in enumerate(client_ids):
                    weight = weights[idx].item()
                    if idx == 0:
                        aggregated_weights = {k: v.clone() * weight for k, v in client_weights[client_id].items()}
                    else:
                        for k in aggregated_weights.keys():
                            aggregated_weights[k] += client_weights[client_id][k] * weight
            else:
                # 选取距离和最小的客户端
                best_idx = torch.argmin(scores).item()
                best_client_id = client_ids[best_idx]
                print(f"Krum选择客户端 {best_client_id} 作为距离最优的更新")
                aggregated_weights = client_weights[best_client_id]
        else:  # MultiKrum
            # MultiKrum：选取得分最低的 k 个客户端，然后平均
            k = min(self.args.multikrum_k, num_clients)
            top_k_indices = torch.topk(scores, k, largest=False).indices
            selected_client_ids = [client_ids[idx.item()] for idx in top_k_indices]
            print(f"MultiKrum选择客户端 {selected_client_ids} 作为距离最优的 {k} 个更新")
            
            # 对选出的客户端更新进行平均
            total_selected = 0
            aggregated_weights = None
            
            for idx in top_k_indices:
                client_id = client_ids[idx.item()]
                client_weight = updates[client_id][1]  # 获取客户端数据大小
                
                # 初始化或累加权重
                if aggregated_weights is None:
                    aggregated_weights = {k: v.clone() * client_weight for k, v in client_weights[client_id].items()}
                    total_selected = client_weight
                else:
                    for k in aggregated_weights.keys():
                        aggregated_weights[k] += client_weights[client_id][k] * client_weight
                        total_selected += client_weight
            
            # 计算平均值
            if total_selected > 0:
                for k in aggregated_weights.keys():
                    aggregated_weights[k] = aggregated_weights[k] / total_selected
                    
        return aggregated_weights
    
    def bulyan_aggregate(self, updates):
        """
        Bulyan聚合算法实现:
        1. 首先运行多次Krum选择可信客户端集合
        2. 然后对选出的客户端模型参数进行逐坐标修剪平均
        """
        # 提取所有客户端的更新
        client_weights = {}
        client_sample_sizes = {}
        
        for client_id, (weights, num_samples, _) in updates.items():
            client_weights[client_id] = weights
            client_sample_sizes[client_id] = num_samples
        
        # 转换模型参数为向量表示，用于计算距离
        weight_vectors = {}
        param_shapes = {}  # 记录每个参数的形状，用于后续重构
        flattened_weights = {}  # 存储每个客户端的平铺参数，用于逐坐标操作
        
        for client_id, weights in client_weights.items():
            # 创建参数字典形状的副本（如果还没有）
            if not param_shapes:
                for name, param in weights.items():
                    param_shapes[name] = param.shape
            
            # 平铺所有参数
            vector = []
            flat_weights = {}
            
            for name, param in weights.items():
                flat_param = param.view(-1)
                vector.append(flat_param)
                flat_weights[name] = flat_param
                
            weight_vectors[client_id] = torch.cat(vector)
            flattened_weights[client_id] = flat_weights
        
        # 计算客户端模型两两之间的欧氏距离
        num_clients = len(weight_vectors)
        distances = torch.zeros(num_clients, num_clients)
        client_ids = list(weight_vectors.keys())
        
        for i in range(num_clients):
            for j in range(i+1, num_clients):
                client_i = client_ids[i]
                client_j = client_ids[j]
                dist = torch.norm(weight_vectors[client_i] - weight_vectors[client_j])
                distances[i, j] = dist
                distances[j, i] = dist
        
        # 计算krum得分（第一阶段的筛选）
        n = num_clients
        f = min(self.args.num_malicious_tolerance, n-2)  # 容忍 f 个拜占庭错误
        
        if f >= n/2:
            print(f"警告: Bulyan要求拜占庭错误少于一半客户端 (f < n/2)。当前 f={f}, n={n}, 回退到MultiKrum。")
            # 如果无法满足Bulyan要求，回退到MultiKrum
            self.args.fed_algo = "MultiKrum"
            return self.krum_aggregate(updates)
        
        m = n - f - 2  # 用于Krum得分计算
        
        if m <= 0:
            print(f"警告: 客户端数量不足，无法应用Bulyan算法。至少需要 {f+3} 个客户端，但只有 {n} 个。回退到FedAvg。")
            # 回退到FedAvg
            return self.fedavg_aggregate(updates)
        
        # 第一阶段: 类似MultiKrum，但必须选择至少2f+3个客户端
        # 计算客户端的Krum得分
        scores = torch.zeros(num_clients)
        
        for i in range(num_clients):
            # 获取与当前客户端距离最近的 m 个客户端的距离
            client_distances = distances[i].clone()
            client_distances[i] = float('inf')  # 排除自己
            closest_m_distances, _ = torch.topk(client_distances, m, largest=False)
            # Krum得分是与最近的 m 个客户端的距离总和
            scores[i] = torch.sum(closest_m_distances)
        
        # 选择至少2f+3个客户端（Bulyan要求的最小安全数量）
        needed_size = 2 * f + 3
        selected_size = min(max(self.args.multikrum_k, needed_size), num_clients)
        
        print(f"Bulyan第一阶段: 选择 {selected_size} 个客户端进行逐坐标修剪，要求至少 {needed_size} 个客户端")
        
        # 选择得分最低的选定数量客户端
        top_indices = torch.topk(scores, selected_size, largest=False).indices
        selected_client_ids = [client_ids[idx.item()] for idx in top_indices]
        
        print(f"Bulyan第一阶段选择客户端: {selected_client_ids}")
        
        # 第二阶段: 对选出的客户端进行逐坐标修剪平均
        # 创建结果字典
        aggregated_weights = {}
        
        # 修剪数量 (Beta参数控制修剪程度)
        beta = self.args.bulyan_beta
        trim_size = int(min(f, selected_size / 4) * beta)  # 修剪两端
        trim_size = max(1, min(trim_size, (selected_size - 1) // 2))  # 至少保留一半客户端
        
        print(f"Bulyan第二阶段: 每个坐标去除 {trim_size} 个最大和最小值")
        
        # 对每个参数层进行操作
        for param_name in param_shapes.keys():
            # 收集所有选中客户端的此参数
            selected_params = []
            selected_weights = []
            
            for idx in top_indices:
                client_id = client_ids[idx.item()]
                selected_params.append(flattened_weights[client_id][param_name])
                selected_weights.append(client_sample_sizes[client_id])
            
            # 转换为张量形式，便于操作
            # [num_selected, param_size]
            params_tensor = torch.stack(selected_params) 
            
            # 计算权重归一化因子
            total_weight = sum(selected_weights)
            if total_weight > 0:
                norm_weights = [w / total_weight for w in selected_weights]
            else:
                norm_weights = [1.0 / len(selected_weights)] * len(selected_weights)
                
            # 权重张量: [num_selected, 1]
            weights_tensor = torch.tensor(norm_weights).view(-1, 1).to(self.device)
            
            # 逐坐标操作
            param_size = params_tensor.size(1)
            trimmed_mean = torch.zeros(param_size, device=self.device)
            
            for i in range(param_size):
                # 获取当前坐标的所有值: [num_selected]
                coord_values = params_tensor[:, i]
                
                # 排序找到要保留的中间索引
                sorted_indices = torch.argsort(coord_values)
                
                # 修剪两端的值
                kept_indices = sorted_indices[trim_size:len(sorted_indices) - trim_size]
                
                # 计算修剪平均值（带权重）
                kept_values = coord_values[kept_indices]
                kept_weights = weights_tensor[kept_indices]
                
                # 加权平均: [1]
                trimmed_mean[i] = torch.sum(kept_values * kept_weights)
            
            # 重构为原始形状
            aggregated_weights[param_name] = trimmed_mean.reshape(param_shapes[param_name])
        
        return aggregated_weights
    
    def fedavg_aggregate(self, updates):
        """标准FedAvg聚合，用作回退方法"""
        total_weight = 0
        aggregated_weights = None
        
        for client_id, (weights, num_samples, _) in updates.items():
            total_weight += num_samples
            if aggregated_weights is None:
                aggregated_weights = {k: v.clone() * num_samples for k, v in weights.items()}
            else:
                for k in weights.keys():
                    aggregated_weights[k] += weights[k] * num_samples
        
        # 计算加权平均
        for k in aggregated_weights.keys():
            aggregated_weights[k] = aggregated_weights[k] / total_weight
        
        return aggregated_weights
    
    def trimmed_mean_aggregate(self, updates):
        """
        修剪均值（TrimmedMean）聚合算法:
        对每个参数的每个坐标，去掉最大和最小的若干值后平均
        
        参数:
        - updates: 客户端更新字典，格式为 {client_id: (weights, num_samples, loss)}
        
        返回:
        - aggregated_weights: 聚合后的模型参数
        """
        # 提取所有客户端的更新
        client_weights = {}
        client_sample_sizes = {}
        
        for client_id, (weights, num_samples, _) in updates.items():
            client_weights[client_id] = weights
            client_sample_sizes[client_id] = num_samples
        
        # 获取参数形状信息
        param_shapes = {}
        flattened_weights = {}  # 存储每个客户端的平铺参数，用于逐坐标操作
        
        for client_id, weights in client_weights.items():
            # 创建参数字典形状的副本（如果还没有）
            if not param_shapes:
                for name, param in weights.items():
                    param_shapes[name] = param.shape
            
            # 为每个参数创建平铺表示
            flat_weights = {}
            for name, param in weights.items():
                flat_weights[name] = param.view(-1)
                
            flattened_weights[client_id] = flat_weights
        
        # 计算要修剪的数量
        num_clients = len(client_weights)
        
        # 确定修剪数量
        if hasattr(self.args, 'trim_k') and self.args.trim_k is not None:
            # 直接使用指定的修剪数量
            trim_k = self.args.trim_k
        else:
            # 使用修剪比例计算修剪数量
            trimmed_ratio = getattr(self.args, 'trimmed_ratio', 0.2)
            trim_k = max(1, int(num_clients * trimmed_ratio))
        
        # 确保修剪数量合理
        trim_k = min(trim_k, (num_clients - 1) // 2)  # 最多修剪掉一半（向下取整）
        
        if trim_k <= 0:
            print(f"警告: 客户端数量不足，无法进行修剪。至少需要3个客户端，但只有{num_clients}个。回退到FedAvg。")
            return self.fedavg_aggregate(updates)
        
        print(f"TrimmedMean: 每个坐标修剪 {trim_k} 个最大和最小值 (共 {num_clients} 个客户端)")
        
        # 创建结果字典
        aggregated_weights = {}
        
        # 对每个参数层进行操作
        for param_name in param_shapes.keys():
            # 收集所有客户端的此参数
            param_values = []
            param_weights = []
            
            for client_id in client_weights.keys():
                param_values.append(flattened_weights[client_id][param_name])
                param_weights.append(client_sample_sizes[client_id])
            
            # 转换为张量，便于操作 [num_clients, param_size]
            params_tensor = torch.stack(param_values)
            
            # 计算归一化权重
            total_weight = sum(param_weights)
            if total_weight > 0:
                norm_weights = [w / total_weight for w in param_weights]
            else:
                norm_weights = [1.0 / num_clients] * num_clients
            
            # 权重张量 [num_clients, 1]
            weights_tensor = torch.tensor(norm_weights).view(-1, 1).to(self.device)
            
            # 逐坐标修剪平均
            param_size = params_tensor.size(1)
            trimmed_mean = torch.zeros(param_size, device=self.device)
            
            for i in range(param_size):
                # 获取当前坐标的所有值 [num_clients]
                coord_values = params_tensor[:, i]
                
                # 排序找出要保留的中间索引
                sorted_indices = torch.argsort(coord_values)
                
                # 裁剪两端极值
                kept_indices = sorted_indices[trim_k:num_clients-trim_k]
                
                # 计算修剪平均值（考虑权重）
                kept_values = coord_values[kept_indices]
                kept_weights = weights_tensor[kept_indices]
                
                # 重新归一化权重
                kept_weights = kept_weights / kept_weights.sum()
                
                # 加权平均 [1]
                trimmed_mean[i] = torch.sum(kept_values * kept_weights)
            
            # 重构为原始形状
            aggregated_weights[param_name] = trimmed_mean.reshape(param_shapes[param_name])
        
        return aggregated_weights
    
    def median_aggregate(self, updates):
        """
        中位数（Median）聚合算法:
        对每个参数的每个坐标，取所有客户端值的中位数
        这是一种特殊的修剪均值，等价于裁剪到只剩中间一个值
        
        参数:
        - updates: 客户端更新字典，格式为 {client_id: (weights, num_samples, loss)}
        
        返回:
        - aggregated_weights: 聚合后的模型参数
        """
        # 提取所有客户端的更新
        client_weights = {}
        
        for client_id, (weights, num_samples, _) in updates.items():
            client_weights[client_id] = weights
        
        # 获取参数形状信息
        param_shapes = {}
        flattened_weights = {}  # 存储每个客户端的平铺参数，用于逐坐标操作
        
        for client_id, weights in client_weights.items():
            # 创建参数字典形状的副本（如果还没有）
            if not param_shapes:
                for name, param in weights.items():
                    param_shapes[name] = param.shape
            
            # 为每个参数创建平铺表示
            flat_weights = {}
            for name, param in weights.items():
                flat_weights[name] = param.view(-1)
                
            flattened_weights[client_id] = flat_weights
        
        num_clients = len(client_weights)
        if num_clients < 3:
            print(f"警告: 客户端数量不足，无法计算有意义的中位数。至少需要3个客户端，但只有{num_clients}个。回退到FedAvg。")
            return self.fedavg_aggregate(updates)
        
        print(f"Median: 对每个坐标计算 {num_clients} 个值的中位数")
        
        # 创建结果字典
        aggregated_weights = {}
        
        # 对每个参数层进行操作
        for param_name in param_shapes.keys():
            # 收集所有客户端的此参数
            param_values = []
            
            for client_id in client_weights.keys():
                param_values.append(flattened_weights[client_id][param_name])
            
            # 转换为张量，便于操作 [num_clients, param_size]
            params_tensor = torch.stack(param_values)
            
            # 逐坐标计算中位数
            param_size = params_tensor.size(1)
            median_values = torch.zeros(param_size, device=self.device)
            
            for i in range(param_size):
                # 获取当前坐标的所有值 [num_clients]
                coord_values = params_tensor[:, i]
                
                # 计算中位数
                median_values[i] = torch.median(coord_values)
            
            # 重构为原始形状
            aggregated_weights[param_name] = median_values.reshape(param_shapes[param_name])
        
        return aggregated_weights
    
    def auror_aggregate(self, updates):
        """
        Auror聚合算法:
        通过对客户端模型更新进行聚类分析，识别并过滤可能的恶意更新
        
        参数:
        - updates: 客户端更新字典，格式为 {client_id: (weights, num_samples, loss)}
        
        返回:
        - aggregated_weights: 聚合后的模型参数
        """
        # 提取所有客户端的更新
        client_weights = {}
        client_sample_sizes = {}
        
        for client_id, (weights, num_samples, _) in updates.items():
            client_weights[client_id] = weights
            client_sample_sizes[client_id] = num_samples
        
        # 转换模型参数为向量表示，用于聚类分析
        weight_vectors = {}
        client_ids = []
        
        for client_id, weights in client_weights.items():
            vector = []
            for name, param in weights.items():
                vector.append(param.view(-1))
            weight_vectors[client_id] = torch.cat(vector).cpu().numpy()
            client_ids.append(client_id)
        
        # 将更新向量堆叠为矩阵，每行代表一个客户端的更新
        X = np.array([weight_vectors[client_id] for client_id in client_ids])
        
        # 客户端数量检查
        num_clients = len(client_ids)
        if num_clients < 3:
            print(f"警告: 客户端数量不足，无法进行有效聚类。至少需要3个客户端，但只有{num_clients}个。回退到FedAvg。")
            return self.fedavg_aggregate(updates)
        
        # 确定聚类数量，默认为2（异常簇和正常簇）
        n_clusters = min(self.args.auror_n_clusters, num_clients - 1)
        
        print(f"Auror: 对 {num_clients} 个客户端更新进行聚类分析，聚类数量={n_clusters}")
        
        try:
            # 执行K-means聚类
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=self.args.i_seed).fit(X)
            labels = kmeans.labels_
            centers = kmeans.cluster_centers_
            
            # 计算每个簇的大小
            cluster_sizes = {}
            for label in labels:
                if label not in cluster_sizes:
                    cluster_sizes[label] = 0
                cluster_sizes[label] += 1
            
            # 计算簇之间的距离
            cluster_distances = pairwise_distances(centers)
            
            # 标识可能的恶意簇
            malicious_clusters = set()
            for i in range(n_clusters):
                # 计算簇大小比例
                size_ratio = cluster_sizes[i] / num_clients
                
                # 判断是否是小簇（可能的恶意簇）
                is_small_cluster = size_ratio < self.args.auror_size_threshold
                
                # 计算与其他簇的最小距离
                other_clusters = [j for j in range(n_clusters) if j != i]
                if other_clusters:
                    min_distance = min(cluster_distances[i][j] for j in other_clusters)
                    # 判断是否距离远离其他簇（可能的恶意簇）
                    is_distant_cluster = min_distance > self.args.auror_distance_threshold
                else:
                    is_distant_cluster = False
                
                # 如果簇较小且距离较远，则标记为可能的恶意簇
                if is_small_cluster and is_distant_cluster:
                    malicious_clusters.add(i)
                    print(f"Auror检测到可能的恶意簇: 簇{i}, 大小={cluster_sizes[i]}/{num_clients}, 比例={size_ratio:.2f}, 最小距离={min_distance:.4f}")
            
            # 打印聚类结果
            for i in range(n_clusters):
                client_in_cluster = [client_ids[j] for j in range(len(labels)) if labels[j] == i]
                print(f"簇{i}: 大小={cluster_sizes[i]}, 客户端={client_in_cluster}")
            
            # 收集未被标记为恶意的客户端
            benign_client_ids = []
            for i, label in enumerate(labels):
                if label not in malicious_clusters:
                    benign_client_ids.append(client_ids[i])
            
            print(f"Auror保留 {len(benign_client_ids)}/{num_clients} 个客户端的更新进行聚合")
            
            # 如果所有簇都被标记为恶意（不太可能），回退到使用所有客户端
            if not benign_client_ids:
                print("警告: 所有簇都被标记为恶意，回退到使用所有客户端")
                benign_client_ids = client_ids
            
            # 使用良性客户端的更新进行聚合
            total_weight = 0
            aggregated_weights = None
            
            for client_id in benign_client_ids:
                client_weight = client_sample_sizes[client_id]
                total_weight += client_weight
                if aggregated_weights is None:
                    aggregated_weights = {k: v.clone() * client_weight for k, v in client_weights[client_id].items()}
                else:
                    for k in aggregated_weights.keys():
                        aggregated_weights[k] += client_weights[client_id][k] * client_weight
            
            # 计算加权平均
            if total_weight > 0:
                for k in aggregated_weights.keys():
                    aggregated_weights[k] = aggregated_weights[k] / total_weight
                    
        except Exception as e:
            print(f"Auror聚类过程中出错: {e}。回退到FedAvg。")
            return self.fedavg_aggregate(updates)
        
        return aggregated_weights
    
    def evaluate(self):
        """评估服务器模型性能"""
        self.server_model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                outputs = self.server_model(data)
                _, predicted = torch.max(outputs.data, 1)
                total += target.size(0)
                correct += (predicted == target).sum().item()
                
                # 收集预测和目标值用于计算F1-score
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(target.cpu().numpy())
        
        # 计算F1-score (多分类使用macro平均)
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return correct / total, f1
    
    def select_clients(self):
        """根据客户端比例随机选择参与训练的客户端"""
        all_clients = self.trainset_config['users']
        num_selected = max(1, int(len(all_clients) * self.args.client_ratio))
        selected_clients = random.sample(all_clients, num_selected)
        return selected_clients
    
    def train(self):
        """联邦学习训练过程"""
        pbar = tqdm(range(self.args.num_round))
        max_accuracy = 0
        max_f1 = 0
        
        for round_idx in pbar:
            # 选择参与本轮训练的客户端
            selected_clients = self.select_clients()
            
            # 客户端本地训练
            client_updates = {}
            for client_id in selected_clients:
                # 更新客户端模型
                self.clients[client_id]['model'].load_state_dict(self.server_model.state_dict())
                # 本地训练
                weights, num_samples, loss = self.client_update(client_id)
                client_updates[client_id] = (weights, num_samples, loss)
            
            # 服务器聚合
            aggregated_weights = self.server_aggregate(client_updates)
            self.server_model.load_state_dict(aggregated_weights)
            
            # 评估
            accuracy, f1 = self.evaluate()
            avg_loss = sum(update[2] for update in client_updates.values()) / len(client_updates)
            
            # 记录结果
            self.results['server']['accuracy'].append(accuracy)
            self.results['server']['train_loss'].append(avg_loss)
            self.results['server']['f1_score'].append(f1)
            
            if accuracy > max_accuracy:
                max_accuracy = accuracy
            if f1 > max_f1:
                max_f1 = f1
            
            pbar.set_description(
                f'轮次: {round_idx} | '
                f'算法: {self.args.fed_algo} | '
                f'选中: {len(selected_clients)}/{len(self.trainset_config["users"])} | '
                f'训练损失: {avg_loss:.4f} | '
                f'准确率: {accuracy:.4f} | '
                f'F1-score: {f1:.4f} | '
                f'最佳准确率: {max_accuracy:.4f} | '
                f'最佳F1: {max_f1:.4f}'
            )
            
            # 保存结果
            if not os.path.exists(self.args.res_root):
                os.makedirs(self.args.res_root)
            
            # 添加攻击信息到文件名
            attack_suffix = ""
            if hasattr(self.args, 'enable_attack') and self.args.enable_attack:
                attack_suffix = f"_attack_{self.args.attack_type}_{self.args.num_malicious}_{self.args.noise_level}"
            
            # 添加alpha参数信息，标识non-iid程度
            alpha_suffix = ""
            if self.args.distribution_type == 'non_iid_label' and hasattr(self.args, 'alpha'):
                alpha_suffix = f"_alpha{self.args.alpha}"
            
            result_path = os.path.join(
                self.args.res_root,
                f'[{self.args.fed_algo}_{self.args.model}_{self.args.distribution_type}{alpha_suffix}_{self.args.i_seed}{attack_suffix}]'
            )
            with open(result_path, 'w') as f:
                json.dump(self.results, f, cls=PythonObjectEncoder)

def main():
    args = get_args()
    federated_learning = FederatedLearning(args)
    federated_learning.train()

if __name__ == "__main__":
    main() 