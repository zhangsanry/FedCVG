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
from federated_learning import FederatedLearning
from preprocessing.baselines_dataloader import load_data, divide_data
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}

class AurorComm(FederatedLearning):
    def __init__(self, args):
        # 将参数强制设为CIFAR10和ResNet34
        args.dataset = 'CIFAR10'
        args.model = 'ResNet34'
        
        # 确保使用Auror算法
        args.fed_algo = "Auror"
        super().__init__(args)
        
        # 初始化通信开销跟踪
        self.communication_costs = {
            'server_to_client': [],  # 服务器到客户端的通信开销
            'client_to_server': [],  # 客户端到服务器的通信开销
            'total_per_round': [],   # 每轮的总通信开销
            'cumulative': [],        # 累积通信开销
            'num_filtered_clients': []  # 每轮被Auror过滤掉的客户端数量
        }
        self.total_communication_cost = 0
        
        # 创建结果目录
        self.comm_results_dir = os.path.join('results', 'consumption')
        os.makedirs(self.comm_results_dir, exist_ok=True)
    
    def calculate_model_size(self, model_state):
        """计算模型状态字典的大小（MB）"""
        total_bytes = 0
        for param in model_state.values():
            # 计算参数的字节数
            num_elements = param.numel()
            element_size = param.element_size()
            param_bytes = num_elements * element_size
            total_bytes += param_bytes
        
        # 转换为MB
        total_mb = total_bytes / (1024 * 1024)
        return total_mb
    
    def auror_aggregate_with_tracking(self, updates):
        """
        重写Auror聚合算法，添加被过滤客户端数量的跟踪
        
        参数:
        - updates: 客户端更新字典，格式为 {client_id: (weights, num_samples, loss)}
        
        返回:
        - aggregated_weights: 聚合后的模型参数
        - num_filtered: 被过滤掉的客户端数量
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
            return self.fedavg_aggregate(updates), 0
        
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
            
            # 计算被过滤的客户端数量
            num_filtered = num_clients - len(benign_client_ids)
            print(f"Auror保留 {len(benign_client_ids)}/{num_clients} 个客户端的更新进行聚合")
            
            # 如果所有簇都被标记为恶意（不太可能），回退到使用所有客户端
            if not benign_client_ids:
                print("警告: 所有簇都被标记为恶意，回退到使用所有客户端")
                benign_client_ids = client_ids
                num_filtered = 0
            
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
            return self.fedavg_aggregate(updates), 0
        
        return aggregated_weights, num_filtered
    
    def train(self):
        """Training process with communication cost calculation"""
        print(f"\nStarting Auror algorithm training (tracking communication costs)...")
        print(f"Dataset: {self.args.dataset}, Model: {self.args.model}")
        print(f"Total rounds: {self.args.num_round}, Clients: {self.args.num_client}, Participation ratio: {self.args.client_ratio}")
        
        # Get server model size (in MB)
        server_model_size = self.calculate_model_size(self.server_model.state_dict())
        print(f"Server model size: {server_model_size:.4f} MB")
        
        # Calculate total number of model parameters
        total_params = sum(p.numel() for p in self.server_model.parameters())
        print(f"Total model parameters: {total_params}")
        
        # Training loop
        for round_idx in tqdm(range(1, self.args.num_round + 1), desc="Training progress"):
            selected_clients = self.select_clients()
            num_selected = len(selected_clients)
            
            # Record communication cost for distributing model: server -> all selected clients
            server_to_clients_cost = server_model_size * num_selected
            
            # Client updates for this round
            round_client_costs = 0
            updates = {}
            
            for client_id in selected_clients:
                # Copy server model to client
                self.clients[client_id]['model'].load_state_dict(self.server_model.state_dict())
                
                # Client local training
                weights, num_samples, loss = self.client_update(client_id)
                updates[client_id] = (weights, num_samples, loss)
                
                # Calculate communication cost from client to server
                client_model_size = self.calculate_model_size(weights)
                round_client_costs += client_model_size
            
            # Server aggregates model using Auror with malicious client detection
            # The modified aggregate function returns both weights and number of filtered clients
            aggregated_weights, num_filtered = self.auror_aggregate_with_tracking(updates)
            self.server_model.load_state_dict(aggregated_weights)
            
            # Evaluate current model performance
            accuracy, f1 = self.evaluate()
            
            # Record performance metrics
            self.results['server']['accuracy'].append(accuracy)
            self.results['server']['f1_score'].append(f1)
            self.results['server']['train_loss'].append(0)  # Server has no training loss
            
            # Record communication costs
            self.communication_costs['server_to_client'].append(server_to_clients_cost)
            self.communication_costs['client_to_server'].append(round_client_costs)
            round_total_cost = server_to_clients_cost + round_client_costs
            self.communication_costs['total_per_round'].append(round_total_cost)
            self.communication_costs['num_filtered_clients'].append(num_filtered)
            
            self.total_communication_cost += round_total_cost
            self.communication_costs['cumulative'].append(self.total_communication_cost)
            
            # Output results every 10 rounds or on the final round
            if round_idx % 10 == 0 or round_idx == self.args.num_round:
                print(f"\nRound {round_idx}/{self.args.num_round}:")
                print(f"- Accuracy: {accuracy:.4f}")
                print(f"- F1 Score: {f1:.4f}")
                print(f"- Filtered clients: {num_filtered}/{num_selected}")
                print(f"- This round communication cost: {round_total_cost:.4f} MB")
                print(f"- Cumulative communication cost: {self.total_communication_cost:.4f} MB")
        
        # Save communication cost results
        self.save_communication_results()
        
        # Save training results
        self.save_results()
        
        return self.results
    
    def save_communication_results(self):
        """Save communication cost results"""
        results_file = os.path.join(
            self.comm_results_dir, 
            f'Auror_comm_{self.args.dataset}_{self.args.model}_{self.args.num_client}clients.json'
        )
        
        # Create results dictionary
        comm_results = {
            'args': vars(self.args),
            'communication_costs': self.communication_costs,
            'final_accuracy': self.results['server']['accuracy'][-1],
            'final_f1_score': self.results['server']['f1_score'][-1],
            'total_communication_cost_MB': self.total_communication_cost
        }
        
        # Save results
        with open(results_file, 'w') as f:
            json.dump(comm_results, f, indent=4, cls=PythonObjectEncoder)
        
        print(f"\nCommunication cost results saved to: {results_file}")

def main():
    args = get_args()
    auror_comm = AurorComm(args)
    auror_comm.train()

if __name__ == "__main__":
    main() 