import os
import random
import json
import pickle
import torch
import numpy as np
from tqdm import tqdm
from json import JSONEncoder
from federated_parser import get_fedhyb_args
from torch.utils.data import DataLoader
from preprocessing.baselines_dataloader import load_data, divide_data
from sklearn.metrics import f1_score
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances
import torch.optim as optim
from fedhyb import FedHyb

class PythonObjectEncoder(JSONEncoder):    
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return super().default(self, obj)
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}

class FedHybComm(FedHyb):
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
            'cumulative': [],        # Cumulative communication cost
            'phase': []              # Phase of each round
        }
        self.total_communication_cost = 0
        
        # Initialize current round and phase
        self.current_round = 0
        self.current_phase = 1
        
        # Create results directory
        self.comm_results_dir = os.path.join('results', 'consumption')
        os.makedirs(self.comm_results_dir, exist_ok=True)
    
    def calculate_model_size(self, model_state):
        """Calculate the size of model state dictionary (MB)"""
        total_bytes = 0
        for param_name, param in model_state.items():
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
        print(f"\nStarting FedHyb algorithm training (tracking communication costs)...")
        print(f"Dataset: {self.args.dataset}, Model: {self.args.model}")
        print(f"Phase 1 rounds: {self.args.phase1_rounds}, Phase 2 rounds: {self.args.phase2_rounds}")
        print(f"Number of clients: {self.args.num_client}")
        print(f"Phase 1 client ratio: {self.args.phase1_client_ratio}, Phase 2 client ratio: {self.args.phase2_client_ratio}")
        
        # Get full model size (in MB)
        full_model_size = self.calculate_model_size(self.server_model.state_dict())
        print(f"Full model size: {full_model_size:.4f} MB")
        
        # Get size of only classifier layers for phase 2 (from a typical client)
        example_client = next(iter(self.clients.values()))
        classifier_params = {}
        with torch.no_grad():
            # Temporarily freeze feature extraction layers to identify classifier parameters
            trainable_params = self.freeze_feature_layers(example_client['model'], phase=2)
            for name, param in example_client['model'].named_parameters():
                if param.requires_grad:  # Trainable parameters are classifier layer parameters
                    classifier_params[name] = param.clone()
        
        classifier_size = self.calculate_model_size(classifier_params)
        print(f"Classifier layer model size: {classifier_size:.4f} MB")
        
        # Calculate parameter counts and ratios
        # Corrected calculation method, ensuring denominator is all parameters, not just requires_grad=True parameters
        full_model_params = sum(p.numel() for p in self.server_model.parameters())
        
        # Temporarily freeze feature extraction layers to count classifier parameters
        temp_model = self.create_model()
        self.freeze_feature_layers(temp_model, phase=2)
        classifier_params_count = sum(p.numel() for p in temp_model.parameters() if p.requires_grad)
        
        print(f"Total parameters: {full_model_params}")
        print(f"Classifier layer parameters: {classifier_params_count}")
        print(f"Classifier layer ratio: {classifier_params_count/full_model_params*100:.2f}%")
        print(f"Communication advantage: {(full_model_size - classifier_size)/full_model_size*100:.2f}%")
        
        # Phase 1: FedProx or SCAFFOLD training
        print("\n---Phase 1: Feature Learning---")
        for round_idx in tqdm(range(1, self.args.phase1_rounds + 1), desc="Phase 1 progress"):
            self.current_round = round_idx
            selected_clients = self.select_clients(self.args.phase1_client_ratio)
            num_selected = len(selected_clients)
            
            # Record communication cost for distributing model: server -> all selected clients
            server_to_clients_cost = full_model_size * num_selected
            
            # Client updates for this round
            round_client_costs = 0
            updates = {}
            
            for client_id in selected_clients:
                # Copy server model to client
                self.clients[client_id]['model'].load_state_dict(self.server_model.state_dict())
                
                # Client local training (phase 1)
                weights, num_samples, loss = self.client_update(client_id, phase=1)
                updates[client_id] = (weights, num_samples, loss)
                
                # Calculate communication cost from client to server
                client_model_size = self.calculate_model_size(weights)
                round_client_costs += client_model_size
            
            # Detect and handle malicious clients
            if (self.use_clustering or self.use_reputation) and len(updates) > 2:
                malicious_clients = self.detect_malicious_clients(updates)
                if self.use_reputation:
                    self.update_reputation(malicious_clients)
                # Remove updates from identified malicious clients
                for client_id in malicious_clients:
                    if client_id in updates:
                        del updates[client_id]
            
            # Server aggregates model
            aggregated_weights = self.server_aggregate(updates)
            self.server_model.load_state_dict(aggregated_weights)
            
            # Evaluate current model performance
            accuracy, f1 = self.evaluate()
            
            # Record performance metrics
            self.results['server']['accuracy'].append(accuracy)
            self.results['server']['f1_score'].append(f1)
            self.results['server']['train_loss'].append(0)
            self.results['server']['phase'].append(1)
            
            # Record communication costs
            self.communication_costs['server_to_client'].append(server_to_clients_cost)
            self.communication_costs['client_to_server'].append(round_client_costs)
            round_total_cost = server_to_clients_cost + round_client_costs
            self.communication_costs['total_per_round'].append(round_total_cost)
            self.communication_costs['phase'].append(1)
            
            self.total_communication_cost += round_total_cost
            self.communication_costs['cumulative'].append(self.total_communication_cost)
            
            # Output results every 10 rounds or on the final round
            if round_idx % 10 == 0 or round_idx == self.args.phase1_rounds:
                print(f"\nPhase 1 Round {round_idx}/{self.args.phase1_rounds}:")
                print(f"- Accuracy: {accuracy:.4f}")
                print(f"- F1 Score: {f1:.4f}")
                print(f"- This round communication cost: {round_total_cost:.4f} MB")
                print(f"- Cumulative communication cost: {self.total_communication_cost:.4f} MB")
        
        # Phase 2: Personalized learning
        print("\n---Phase 2: Personalized Model Training---")
        self.current_phase = 2
        
        # Freeze feature extraction layers on all clients
        for client_id in self.clients:
            self.freeze_feature_layers(self.clients[client_id]['model'], phase=2)
        
        # Phased training
        for round_idx in tqdm(range(1, self.args.phase2_rounds + 1), desc="Phase 2 progress"):
            self.current_round = self.args.phase1_rounds + round_idx
            selected_clients = self.select_clients(self.args.phase2_client_ratio)
            num_selected = len(selected_clients)
            
            # Record communication cost for distributing model: server -> all selected clients
            # Phase 2 only requires transmission of classifier layer parameters
            server_to_clients_cost = classifier_size * num_selected
            
            # Client updates for this round
            round_client_costs = 0
            updates = {}
            
            for client_id in selected_clients:
                # Update client model (phase 2 only updates classifier layers)
                self.update_model_partial(
                    self.clients[client_id]['model'], 
                    self.server_model.state_dict(), 
                    phase=2
                )
                
                # Client local training (phase 2)
                weights, num_samples, loss = self.client_update(client_id, phase=2)
                updates[client_id] = (weights, num_samples, loss)
                
                # Calculate communication cost from client to server
                # In phase 2, only classifier layer parameters are transmitted
                client_weights = {}
                for name, param in weights.items():
                    if any(name.startswith(p) for p in classifier_params.keys()):
                        client_weights[name] = param
                
                client_model_size = self.calculate_model_size(client_weights)
                round_client_costs += client_model_size
            
            # Server aggregates model (only classifier layers in phase 2)
            aggregated_weights = self.server_aggregate(updates)
            self.server_model.load_state_dict(aggregated_weights)
            
            # Evaluate current model performance
            accuracy, f1 = self.evaluate()
            
            # Record performance metrics
            self.results['server']['accuracy'].append(accuracy)
            self.results['server']['f1_score'].append(f1)
            self.results['server']['train_loss'].append(0)
            self.results['server']['phase'].append(2)
            
            # Record communication costs
            self.communication_costs['server_to_client'].append(server_to_clients_cost)
            self.communication_costs['client_to_server'].append(round_client_costs)
            round_total_cost = server_to_clients_cost + round_client_costs
            self.communication_costs['total_per_round'].append(round_total_cost)
            self.communication_costs['phase'].append(2)
            
            self.total_communication_cost += round_total_cost
            self.communication_costs['cumulative'].append(self.total_communication_cost)
            
            # Output results every 10 rounds or on the final round
            if round_idx % 10 == 0 or round_idx == self.args.phase2_rounds:
                print(f"\nPhase 2 Round {round_idx}/{self.args.phase2_rounds}:")
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
        # Determine phase 1 algorithm name (for filename)
        phase1_algo = 'SCAFFOLD' if self.args.scaffold_for_phase1 else 'FedProx'
        
        results_file = os.path.join(
            self.comm_results_dir, 
            f'FedHyb_{phase1_algo}_comm_{self.args.dataset}_{self.args.model}_{self.args.num_client}clients.json'
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
    args = get_fedhyb_args()
    fedhyb_comm = FedHybComm(args)
    fedhyb_comm.train()

if __name__ == "__main__":
    main() 