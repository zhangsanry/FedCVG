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
import math

class PythonObjectEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}

class FedAvgComm(FederatedLearning):
    def __init__(self, args):
        # Force parameters to be CIFAR10 and ResNet34
        args.dataset = 'CIFAR10'  
        args.model = 'ResNet34'
        
        super().__init__(args)
        
        # Initialize communication cost tracking
        self.communication_costs = {
            'server_to_client': [],  # Communication cost from server to clients
            'client_to_server': [],  # Communication cost from clients to server
            'total_per_round': [],   # Total communication cost per round
            'cumulative': []         # Cumulative communication cost
        }
        self.total_communication_cost = 0
        
        # Create results directory
        self.comm_results_dir = os.path.join('results', 'consumption')
        os.makedirs(self.comm_results_dir, exist_ok=True)
    
    def calculate_model_size(self, model_state):
        """Calculate the size of model state dictionary (MB)"""
        total_bytes = 0
        for param in model_state.values():
            # Calculate bytes of parameters
            num_elements = param.numel()
            element_size = param.element_size()
            param_bytes = num_elements * element_size
            total_bytes += param_bytes
        
        # Convert to MB
        total_mb = total_bytes / (1024 * 1024)
        return total_mb
    
    def train(self):
        """Training process with communication cost calculation"""
        print(f"\nStarting FedAvg algorithm training (tracking communication costs)...")
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
            
            # Server aggregates model
            self.server_model.load_state_dict(self.server_aggregate(updates))
            
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
            
            self.total_communication_cost += round_total_cost
            self.communication_costs['cumulative'].append(self.total_communication_cost)
            
            # Output results every 10 rounds or on the final round
            if round_idx % 10 == 0 or round_idx == self.args.num_round:
                print(f"\nRound {round_idx}/{self.args.num_round}:")
                print(f"- Accuracy: {accuracy:.4f}")
                print(f"- F1 Score: {f1:.4f}")
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
            f'FedAvg_comm_{self.args.dataset}_{self.args.model}_{self.args.num_client}clients.json'
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
    fedavg_comm = FedAvgComm(args)
    fedavg_comm.train()

if __name__ == "__main__":
    main() 