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

class PythonObjectEncoder(JSONEncoder):    
    def default(self, obj):
        if isinstance(obj, (list, dict, str, int, float, bool, type(None))):
            return super().default(self, obj )
        return {'_python_object': pickle.dumps(obj).decode('latin-1')}
  
class FedCVG:
    def __init__(self, args):
        """Initialize FedCVG algorithm"""
        self.args = args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Set random seed
        np.random.seed(args.i_seed)
        torch.manual_seed(args.i_seed)
        random.seed(args.i_seed)
        
        # Load data
        self.trainset_config, self.testset = divide_data(
            num_client=args.num_client,
            dataset_name=args.dataset,
            i_seed=args.i_seed,
            distribution_type=args.distribution_type,
            alpha=args.alpha
        )
        
        # Initialize clients and server
        self.init_clients()
        self.init_server()
        
        # Initialize results recording
        self.current_phase = 1
        self.results = {
            'server': {
                'accuracy': [],
                'train_loss': [],
                'f1_score': [],
                'phase': []
            }
        }
        
        # New: Initialize data structures for gradient recording and client selection history
        self.client_gradients = {}  # Store client gradients
        self.client_last_selected = {}  # Record when each client was last selected
        for client_id in self.clients:
            self.client_gradients[client_id] = None
            self.client_last_selected[client_id] = 0  # Initialize to round 0
        
        # Initialize gradient decay parameter for each client
        if hasattr(args, 'gradient_decay'):
            self.gradient_decay = args.gradient_decay
        else:
            self.gradient_decay = 0.9  # Default decay rate
        
        # Initialize gradient threshold
        if hasattr(args, 'gradient_threshold'):
            self.gradient_threshold = args.gradient_threshold
        else:
            self.gradient_threshold = 0.01  # Default threshold
        
        # Initialize variables needed for clustering and reputation mechanisms
        self.use_clustering = hasattr(args, 'use_clustering') and args.use_clustering
        self.use_reputation = hasattr(args, 'use_reputation') and args.use_reputation
        
        if self.use_clustering or self.use_reputation:
            # Store the number of times each client is marked as malicious
            self.malicious_counts = {client_id: 0 for client_id in self.clients}
            
            # Store the list of clients currently considered malicious
            self.blacklisted_clients = set()
            
            # Set threshold: number of times marked as malicious exceeds the proportion of phase 1 training rounds
            self.reputation_threshold = args.clustering_threshold * args.phase1_rounds if hasattr(args, 'clustering_threshold') else 0.1 * args.phase1_rounds
            
            print(f"Enabled {'clustering mechanism' if self.use_clustering else ''}{'and' if self.use_clustering and self.use_reputation else ''}{'reputation mechanism' if self.use_reputation else ''}")
            if self.use_reputation:
                print(f"Reputation threshold set to: {self.reputation_threshold:.1f} rounds ({args.clustering_threshold if hasattr(args, 'clustering_threshold') else 0.1:.1%} of phase 1 total rounds)")
        
        # Initialize control variables for SCAFFOLD algorithm (if using SCAFFOLD in phase 1)
        if hasattr(args, 'scaffold_for_phase1') and args.scaffold_for_phase1:
            self.control_variate = {}
            for name, param in self.server_model.state_dict().items():
                self.control_variate[name] = torch.zeros_like(param)
            
            # Initialize control variables for each client
            for client_id in self.clients:
                self.clients[client_id]['control_variate'] = {}
                for name, param in self.server_model.state_dict().items():
                    self.clients[client_id]['control_variate'][name] = torch.zeros_like(param)
    
    def init_clients(self):
        """Initialize all clients"""
        self.clients = {}
        for client_id in self.trainset_config['users']:
            self.clients[client_id] = {
                'model': self.create_model(),
                'optimizer': None,
                'data': self.trainset_config['user_data'][client_id]
            }
    
    def init_server(self):
        """Initialize server"""
        self.server_model = self.create_model()
        self.test_loader = DataLoader(self.testset, batch_size=self.args.batch_size)
    
    def create_model(self):
        """Create model instance"""
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
            raise NotImplementedError(f"Model {self.args.model} not implemented")
    
    def freeze_feature_layers(self, model, phase):
        """
        Freeze or unfreeze feature extraction layers based on training phase and model type
        
        Parameters:
            model: Model instance
            phase: Training phase, 1 for phase 1 (all layers trainable), 2 for phase 2 (freeze feature extraction layers)
        
        Returns:
            trainable_params: List of parameters to train
        """
        # If phase 1, all parameters are trainable
        if phase == 1:
            for param in model.parameters():
                param.requires_grad = True
            return model.parameters()
        
        # In phase 2, freeze feature extraction layers, only train classification layers
        trainable_params = []
        
        # Determine feature extraction and classification layers based on model type
        if isinstance(model, type(self.server_model)):
            model_name = model.__class__.__name__
            
            # Handle different models
            if 'LeNet' in model_name:
                # LeNet: conv1, conv2 are feature extraction layers; fc1, fc2, fc3 are classification layers
                feature_layers = ['conv1', 'conv2']
                classifier_layers = ['fc1', 'fc2', 'fc3']
                
                # Freeze feature extraction layers
                for name, param in model.named_parameters():
                    if any(layer in name for layer in feature_layers):
                        param.requires_grad = False
                    elif any(layer in name for layer in classifier_layers):
                        param.requires_grad = True
                        trainable_params.append(param)
            
            elif 'CNN' in model_name:
                # CNN: conv1, conv2, conv3 are feature extraction layers; fc1, fc2 are classification layers
                feature_layers = ['conv1', 'conv2', 'conv3']
                classifier_layers = ['fc1', 'fc2']
                
                # Freeze feature extraction layers
                for name, param in model.named_parameters():
                    if any(layer in name for layer in feature_layers):
                        param.requires_grad = False
                    elif any(layer in name for layer in classifier_layers):
                        param.requires_grad = True
                        trainable_params.append(param)
            
            elif 'ResNet' in model_name:
                # ResNet: conv1, layer1-4 are feature extraction layers; linear is classification layer
                feature_layers = ['conv1', 'layer1', 'layer2', 'layer3', 'layer4', 'bn']
                classifier_layers = ['linear']
                
                # Freeze feature extraction layers
                for name, param in model.named_parameters():
                    if any(layer in name for layer in feature_layers):
                        param.requires_grad = False
                    elif any(layer in name for layer in classifier_layers):
                        param.requires_grad = True
                        trainable_params.append(param)
            
            elif 'AlexCifarNet' in model_name:
                # AlexNet: features are feature extraction layers; classifier are classification layers
                # Freeze feature extraction layers
                for name, param in model.named_parameters():
                    if 'features' in name:
                        param.requires_grad = False
                    elif 'classifier' in name:
                        param.requires_grad = True
                        trainable_params.append(param)
            
            else:
                # Default case: freeze first 70% of layers (assuming earlier layers are feature extraction)
                all_params = list(model.named_parameters())
                split_idx = int(len(all_params) * 0.7)
                
                for i, (name, param) in enumerate(all_params):
                    if i < split_idx:  # First 70% are feature extraction layers
                        param.requires_grad = False
                    else:  # Last 30% are classification layers
                        param.requires_grad = True
                        trainable_params.append(param)
                        
            # Display classification layer information once when entering phase 2
            if phase == 2 and self.current_phase == 1 and self.current_round == self.args.phase1_rounds - 1:
                model_state = model.state_dict()
                print(f"\nPhase 2 Information: {'='*40}")
                print(f"Model: {model_name}")
                print(f"Frozen feature extraction layers: {sum(1 for name, param in model.named_parameters() if not param.requires_grad)}")
                print(f"Trainable classification layers: {len(trainable_params)}")
                
                # Print classification layer names and dimensions
                print("\nClassification layer dimensions:")
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        print(f"  - {name}: {list(param.size())}")
                print('='*60)
                
            return trainable_params
        else:
            # If model type cannot be identified, all parameters are trainable
            return model.parameters() 

    def client_update(self, client_id, phase=1):
        """
        Client local training
        
        Parameters:
            client_id: Client ID
            phase: Training phase, 1 or 2
        
        Returns:
            model_update: Model parameter update (weight difference)
            loss: Training loss
        """
        client = self.clients[client_id]
        
        # Set model to training mode
        client['model'].train()
        
        # Get data loader for this client
        train_loader = DataLoader(client['data'], batch_size=self.args.batch_size, shuffle=True)
        
        # Set learning rate based on phase
        lr = self.args.learning_rate if phase == 1 else self.args.phase2_learning_rate
        
        # Get trainable parameters based on phase
        trainable_params = self.freeze_feature_layers(client['model'], phase)
        
        # Create optimizer for client model
        client['optimizer'] = optim.SGD(
            trainable_params,
            lr=lr,
            momentum=self.args.momentum,
            weight_decay=self.args.weight_decay
        )
        
        # Initialize loss
        train_loss = 0.0
        
        # Copy server model parameters to client model
        with torch.no_grad():
            for name, param in self.server_model.state_dict().items():
                client['model'].state_dict()[name].copy_(param.clone())
        
        # First phase specific implementations
        if phase == 1:
            # Use SCAFFOLD algorithm
            if hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1:
                return self._client_update_scaffold(client_id, train_loader)
            # Use FedProx algorithm
            else:
                return self._client_update_fedprox(client_id, train_loader)
        
        # Second phase uses FedAvg algorithm by default
        else:
            return self._client_update_fedavg(client_id, train_loader)
    
    def _client_update_fedavg(self, client_id, train_loader):
        """Standard FedAvg algorithm client update"""
        client = self.clients[client_id]
        criterion = torch.nn.CrossEntropyLoss()
        
        # Save initial model parameters for calculating update later
        init_params = {name: param.clone() for name, param in client['model'].state_dict().items()}
        
        # Training for local epochs
        train_loss = 0.0
        for epoch in range(self.args.num_local_epoch):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = client['model'](inputs)
                loss = criterion(outputs, labels)
                
                # Backward and optimize
                client['optimizer'].zero_grad()
                loss.backward()
                client['optimizer'].step()
                
                epoch_loss += loss.item()
            
            train_loss += epoch_loss / len(train_loader)
        
        # Calculate average training loss
        train_loss /= self.args.num_local_epoch
        
        # Calculate model update (weight difference)
        model_update = {}
        for name, param in client['model'].state_dict().items():
            if param.requires_grad:
                model_update[name] = param.clone() - init_params[name]
            else:
                model_update[name] = torch.zeros_like(param)
        
        # Store client gradient for future use
        if hasattr(self.args, 'use_historical_gradients') and self.args.use_historical_gradients:
            self.client_gradients[client_id] = {k: v.clone() for k, v in model_update.items()}
        
        return model_update, train_loss
    
    def _client_update_fedprox(self, client_id, train_loader):
        """FedProx algorithm client update with proximal term"""
        client = self.clients[client_id]
        criterion = torch.nn.CrossEntropyLoss()
        
        # Get proximal term coefficient
        mu = self.args.fedprox_mu
        
        # Save initial model parameters for proximal term and calculating update later
        global_params = {name: param.clone() for name, param in self.server_model.state_dict().items()}
        init_params = {name: param.clone() for name, param in client['model'].state_dict().items()}
        
        # Training for local epochs
        train_loss = 0.0
        for epoch in range(self.args.num_local_epoch):
            epoch_loss = 0.0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = client['model'](inputs)
                
                # FedProx loss includes proximal term
                loss = criterion(outputs, labels)
                
                # Add proximal term to loss
                if mu > 0:
                    proximal_term = 0.0
                    for name, param in client['model'].named_parameters():
                        if param.requires_grad:
                            proximal_term += (mu / 2) * torch.norm((param - global_params[name]))**2
                    
                    loss += proximal_term
                
                # Backward and optimize
                client['optimizer'].zero_grad()
                loss.backward()
                client['optimizer'].step()
                
                epoch_loss += loss.item()
            
            train_loss += epoch_loss / len(train_loader)
        
        # Calculate average training loss
        train_loss /= self.args.num_local_epoch
        
        # Calculate model update (weight difference)
        model_update = {}
        for name, param in client['model'].state_dict().items():
            if param.requires_grad:
                model_update[name] = param.clone() - init_params[name]
            else:
                model_update[name] = torch.zeros_like(param)
        
        # Store client gradient for future use
        if hasattr(self.args, 'use_historical_gradients') and self.args.use_historical_gradients:
            self.client_gradients[client_id] = {k: v.clone() for k, v in model_update.items()}
        
        return model_update, train_loss
    
    def _client_update_scaffold(self, client_id, train_loader):
        """SCAFFOLD algorithm client update with control variates"""
        client = self.clients[client_id]
        criterion = torch.nn.CrossEntropyLoss()
        
        # Get SCAFFOLD control variables learning rate
        c_lr = self.args.scaffold_c_lr if hasattr(self.args, 'scaffold_c_lr') else 0.1 * self.args.learning_rate
        
        # Save initial model parameters for calculating update later
        init_params = {name: param.clone() for name, param in client['model'].state_dict().items()}
        
        # Get server and client control variates
        c_global = self.control_variate
        c_local = client['control_variate']
        
        # Training for local epochs
        train_loss = 0.0
        
        for epoch in range(self.args.num_local_epoch):
            epoch_loss = 0.0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Forward pass
                outputs = client['model'](inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass
                client['optimizer'].zero_grad()
                loss.backward()
                
                # Apply SCAFFOLD control variates adjustment
                for name, param in client['model'].named_parameters():
                    if param.requires_grad and param.grad is not None:
                        # Adjust gradient with control variate difference
                        param.grad.data += c_global[name] - c_local[name]
                
                # Update model parameters
                client['optimizer'].step()
                
                epoch_loss += loss.item()
            
            train_loss += epoch_loss / len(train_loader)
        
        # Calculate average training loss
        train_loss /= self.args.num_local_epoch
        
        # Calculate model update (weight difference)
        model_update = {}
        for name, param in client['model'].state_dict().items():
            if param.requires_grad:
                model_update[name] = param.clone() - init_params[name]
            else:
                model_update[name] = torch.zeros_like(param)
        
        # Update client control variates
        for name, param in client['model'].named_parameters():
            if param.requires_grad:
                # Update local control variate
                c_local_new = c_local[name] - c_global[name] + (init_params[name] - param) / (self.args.num_local_epoch * client['optimizer'].param_groups[0]['lr'])
                client['control_variate'][name] = c_local_new
        
        # Store client gradient for future use
        if hasattr(self.args, 'use_historical_gradients') and self.args.use_historical_gradients:
            self.client_gradients[client_id] = {k: v.clone() for k, v in model_update.items()}
        
        return model_update, train_loss 

    def server_aggregate(self, updates):
        """
        Server aggregates client updates using weighted averaging
        
        Parameters:
            updates: Dictionary of client updates, {client_id: (model_update, loss)}
        
        Returns:
            avg_loss: Average client loss
        """
        n_clients = len(updates)
        if n_clients == 0:
            print("Warning: No client updates received for aggregation")
            return 0.0
        
        # Extract losses and calculate weighted average
        losses = [updates[client_id][1] for client_id in updates]
        avg_loss = sum(losses) / n_clients
        
        # Second phase, if using historical gradients mechanism
        if self.current_phase == 2 and hasattr(self.args, 'use_historical_gradients') and self.args.use_historical_gradients:
            # Use historical gradients for non-selected clients
            # Get list of selected and unselected clients
            selected_clients = list(updates.keys())
            all_clients = list(self.clients.keys())
            # Filter out blacklisted clients
            if hasattr(self, 'blacklisted_clients'):
                unselected_clients = [c for c in all_clients if c not in selected_clients and c not in self.blacklisted_clients]
            else:
                unselected_clients = [c for c in all_clients if c not in selected_clients]
            
            round_factor = self.current_round - self.args.phase1_rounds  # Current round in phase 2
            
            # Use historical gradients only after certain rounds in phase 2
            if round_factor >= 2:  # Start using historical gradients after 2 rounds in phase 2
                updates_with_history = dict(updates)  # Create a copy of updates
                history_count = 0
                
                for client_id in unselected_clients:
                    # Check if client has historical gradients
                    if client_id in self.client_gradients and self.client_gradients[client_id] is not None:
                        # Calculate rounds since last selection
                        rounds_passed = self.current_round - self.client_last_selected[client_id]
                        
                        if rounds_passed > 0:
                            # Decay factor based on rounds passed since last selection
                            decay_factor = max(0, 1.0 - (rounds_passed * self.gradient_decay / 10.0))
                            
                            # Only use historical gradient if decay factor is above threshold
                            if decay_factor > self.gradient_threshold:
                                # Scale historical gradient by decay factor
                                scaled_gradient = {k: v * decay_factor for k, v in self.client_gradients[client_id].items()}
                                
                                # Add to updates with historical gradients
                                updates_with_history[client_id] = (scaled_gradient, avg_loss)  # Use average loss for weighting
                                history_count += 1
                
                if history_count > 0:
                    print(f"Using {history_count} historical client gradients with decay {self.gradient_decay} (round {self.current_round}, phase 2)")
                    # Use updated dictionary with historical gradients
                    updates = updates_with_history
        
        # Extract model updates
        client_updates = {client_id: updates[client_id][0] for client_id in updates}
        
        # Apply parameter updates to server model
        with torch.no_grad():
            if self.current_phase == 1 and hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1:
                # Update server model with SCAFFOLD aggregation
                self._server_aggregate_scaffold(client_updates)
            else:
                # Update server model with FedAvg/FedProx aggregation
                self._server_aggregate_fedavg(client_updates)
        
        return avg_loss
    
    def _server_aggregate_fedavg(self, client_updates):
        """FedAvg/FedProx aggregation method"""
        n_clients = len(client_updates)
        
        # Create average model update dictionary
        avg_update = {}
        for name, param in self.server_model.state_dict().items():
            updates_sum = torch.zeros_like(param)
            for client_id in client_updates:
                updates_sum += client_updates[client_id][name]
            
            # Compute average update
            avg_update[name] = updates_sum / n_clients
            
            # Apply update to server model
            param.add_(avg_update[name])
    
    def _server_aggregate_scaffold(self, client_updates):
        """SCAFFOLD aggregation method with control variates update"""
        n_clients = len(client_updates)
        
        # Create average model update dictionary
        avg_update = {}
        for name, param in self.server_model.state_dict().items():
            updates_sum = torch.zeros_like(param)
            for client_id in client_updates:
                updates_sum += client_updates[client_id][name]
            
            # Compute average update
            avg_update[name] = updates_sum / n_clients
            
            # Apply update to server model
            param.add_(avg_update[name])
        
        # Update server control variates
        for client_id in client_updates:
            for name in self.control_variate:
                # Update global control variate with average of client control variates
                self.control_variate[name] += (self.clients[client_id]['control_variate'][name] - self.control_variate[name]) / n_clients
    
    def evaluate(self):
        """
        Evaluate server model on test data
        
        Returns:
            accuracy: Test accuracy
            f1: F1 score
        """
        self.server_model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.server_model(inputs)
                _, predicted = outputs.max(1)
                
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
                
                # Store predictions and targets for F1 score calculation
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = correct / total
        f1 = f1_score(all_targets, all_preds, average='macro')
        
        return accuracy, f1
    
    def detect_malicious_clients(self, updates):
        """
        Detect potentially malicious clients using clustering
        
        Parameters:
            updates: Dictionary of client updates, {client_id: (model_update, loss)}
        
        Returns:
            malicious_clients: List of potentially malicious client IDs
        """
        # If clustering is not enabled, return empty list
        if not self.use_clustering:
            return []
        
        malicious_clients = []
        
        try:
            # Extract client IDs and model updates
            client_ids = list(updates.keys())
            
            if len(client_ids) <= 2:
                print("Too few clients for clustering analysis")
                return []
            
            # Flatten model updates for each client into 1D vectors for clustering
            flattened_updates = []
            
            for client_id in client_ids:
                # Extract model update
                model_update = updates[client_id][0]
                
                # Flatten update into 1D vector
                flat_update = []
                for name, param in model_update.items():
                    if 'num_batches_tracked' not in name:  # Skip batch norm tracking parameters
                        flat_update.append(param.cpu().reshape(-1))
                
                flat_update = torch.cat(flat_update)
                flattened_updates.append(flat_update.numpy())
            
            # Convert to numpy array for clustering
            X = np.array(flattened_updates)
            
            # Normalize updates to unit norm
            norms = np.linalg.norm(X, axis=1, keepdims=True)
            norms[norms == 0] = 1  # Avoid division by zero
            X_normalized = X / norms
            
            # Use KMeans for clustering
            # Determine number of clusters based on configuration, but ensure reasonable values
            # We want at most num_clients/2 clusters, and at least 2 clusters
            n_clusters = min(max(self.args.fedhyb_cluster_count, 3), len(client_ids) - 1)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto").fit(X_normalized)
            
            # Get cluster labels and calculate cluster sizes
            labels = kmeans.labels_
            cluster_sizes = {}
            for i in range(n_clusters):
                cluster_sizes[i] = np.sum(labels == i)
            
            # Calculate pairwise distances between cluster centers
            centers = kmeans.cluster_centers_
            distances = pairwise_distances(centers)
            
            # For each cluster, find its minimum distance to any other cluster
            min_distances = {}
            for i in range(n_clusters):
                # Set diagonal to infinity to ignore self-distance
                dist_row = distances[i].copy()
                dist_row[i] = float('inf')
                min_distances[i] = np.min(dist_row)
            
            # Identify potential malicious clusters based on size and distance
            malicious_clusters = []
            
            # Threshold for cluster size as a proportion of total clients
            size_threshold = min(self.args.fedhyb_size_threshold, 0.25)
            for i in range(n_clusters):
                size_ratio = cluster_sizes[i] / len(client_ids)
                
                # A small cluster that is far from others may be malicious
                if size_ratio < size_threshold:
                    # Adjust distance threshold based on empirical factors
                    distance_threshold = self.args.fedhyb_distance_threshold * 0.75
                    
                    # If the cluster is small and far from others, mark it as suspicious
                    min_distance = min_distances[i]
                    if min_distance > distance_threshold:
                        malicious_clusters.append(i)
                        print(f"FedCVG clustering detected potential malicious cluster: cluster {i}, size={cluster_sizes[i]}/{len(client_ids)}, ratio={size_ratio:.2f}, min distance={min_distance:.4f}")
            
            # Map cluster indices to client IDs
            for i, label in enumerate(labels):
                if label in malicious_clusters:
                    malicious_clients.append(client_ids[i])
            
            # Report results
            if malicious_clients:
                print(f"FedCVG round {self.current_round+1} detected {len(malicious_clients)} potential malicious clients: {malicious_clients}")
            
            return malicious_clients
            
        except Exception as e:
            print(f"Error during FedCVG clustering analysis: {e}")
            return [] 

    def update_reputation(self, malicious_clients):
        """
        Update reputation scores for clients based on clustering results
        
        Parameters:
            malicious_clients: List of client IDs detected as potentially malicious
            
        Returns:
            blacklisted_clients: Set of client IDs that should be excluded from training
        """
        # If reputation system not enabled, return empty set
        if not self.use_reputation:
            return set()
        
        # Update malicious counts for detected clients
        for client_id in malicious_clients:
            if client_id in self.malicious_counts:
                self.malicious_counts[client_id] += 1
                
                # Check if client should be permanently blacklisted
                if self.malicious_counts[client_id] > self.reputation_threshold:
                    self.blacklisted_clients.add(client_id)
                    print(f"Client {client_id} has been permanently blacklisted (detected malicious {self.malicious_counts[client_id]} times)")
        
        # Report current blacklist status
        if self.blacklisted_clients:
            print(f"Currently blacklisted clients: {self.blacklisted_clients}")
            
        return self.blacklisted_clients
    
    def select_clients(self, client_ratio):
        """
        Select a subset of clients for the current round
        
        Parameters:
            client_ratio: Ratio of clients to select
            
        Returns:
            selected_clients: List of selected client IDs
        """
        # Determine available clients (excluding blacklisted ones)
        available_clients = [c for c in self.clients.keys() if c not in self.blacklisted_clients]
        
        # Calculate number of clients to select
        n_select = max(1, int(client_ratio * len(available_clients)))
        
        # Randomly select clients
        selected_clients = np.random.choice(available_clients, n_select, replace=False).tolist()
        
        # Update last selection round for selected clients
        for client_id in selected_clients:
            self.client_last_selected[client_id] = self.current_round
            
        return selected_clients
    
    def update_model_partial(self, client_model, server_weights, phase):
        """
        Update only a portion of the model weights (for phase 2, only update classification layers)
        
        Parameters:
            client_model: Client model instance
            server_weights: Server model state_dict
            phase: Current training phase
            
        Returns:
            None, updates client_model in-place
        """
        # Determine which parameters to update based on phase
        if phase == 1:
            # In phase 1, update all parameters
            with torch.no_grad():
                for name, param in client_model.state_dict().items():
                    param.copy_(server_weights[name])
        else:
            # In phase 2, only update classification layer parameters
            model_name = client_model.__class__.__name__
            
            # Define classification layer names for different models
            if 'LeNet' in model_name:
                classifier_layers = ['fc1', 'fc2', 'fc3'] 
            elif 'CNN' in model_name:
                classifier_layers = ['fc1', 'fc2']
            elif 'ResNet' in model_name:
                classifier_layers = ['linear']
            elif 'AlexCifarNet' in model_name:
                classifier_layers = ['classifier']
            else:
                # Default: assume last 30% of parameters are classification layers
                all_params = list(client_model.state_dict().keys())
                split_idx = int(len(all_params) * 0.7)
                classifier_layers = all_params[split_idx:]
            
            # Update only classification layer parameters
            with torch.no_grad():
                for name, param in client_model.state_dict().items():
                    if any(layer in name for layer in classifier_layers):
                        param.copy_(server_weights[name])
    
    def save_results(self):
        """Save training results to a file"""
        # Create result filename based on algorithm configuration
        attack_suffix = f"_attack_{self.args.attack_type}_{self.args.num_malicious}" if self.args.enable_attack else ""
        algo_suffix = '_clust' if self.use_clustering else ''
        
        if hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1:
            phase1_algo = 'SCAFFOLD'
        else:
            phase1_algo = 'FedProx'
            
        phase2_suffix = '_hist' if hasattr(self.args, 'use_historical_gradients') and self.args.use_historical_gradients else ''
        
        filename = f'[FedCVG_{phase1_algo}{algo_suffix}{phase2_suffix}_{self.args.model}_non_iid_label_alpha{self.args.alpha}_{self.args.i_seed}]{attack_suffix}'
        
        # Prepare results data for saving
        results_dict = {
            'args': vars(self.args),
            'results': self.results,
            'blacklisted_clients': list(self.blacklisted_clients) if hasattr(self, 'blacklisted_clients') else [],
            'malicious_counts': self.malicious_counts if hasattr(self, 'malicious_counts') else {}
        }
        
        # Create backup directory if it doesn't exist
        backup_path = os.path.join(self.args.res_root, f'fedcvg_results_backup_{self.args.i_seed}')
        if not os.path.exists(backup_path):
            os.makedirs(backup_path)
            
        # Save to backup location first (to prevent data loss in case of crash)
        backup_file = os.path.join(backup_path, f"{filename}.json")
        with open(backup_file, 'w') as f:
            json.dump(results_dict, f, cls=PythonObjectEncoder, indent=4)
            
        # Then save to main results directory
        file_path = os.path.join(self.args.res_root, f"{filename}.json")
        with open(file_path, 'w') as f:
            json.dump(results_dict, f, cls=PythonObjectEncoder, indent=4)
            
        return file_path
    
    def train(self):
        """
        FedCVG training process
        
        Returns:
            results: Dictionary of training results
        """
        # Set up progress tracking
        if hasattr(self.args, 'scaffold_for_phase1') and self.args.scaffold_for_phase1:
            print("=== FedCVG (SCAFFOLD -> FedAvg) Training Started ===")
        else:
            print("=== FedCVG (FedProx -> FedAvg) Training Started ===")
            
        # Track current round and phase
        self.current_round = 0
        self.current_phase = 1
        
        # Phase 1 Training
        print("\nPhase 1 - Feature Extraction Training:")
        for round_idx in range(self.args.phase1_rounds):
            self.current_round = round_idx
            
            # Select clients for this round
            selected_clients = self.select_clients(self.args.phase1_client_ratio)
            
            # Print round info
            print(f"\nRound {round_idx+1}/{self.args.total_rounds} (Phase 1): {len(selected_clients)} clients selected")
            
            # Client update
            client_updates = {}
            for client_id in tqdm(selected_clients, desc="Client Training"):
                # Skip if client in blacklist
                if hasattr(self, 'blacklisted_clients') and client_id in self.blacklisted_clients:
                    continue
                    
                # Perform client update
                update, loss = self.client_update(client_id, phase=1)
                client_updates[client_id] = (update, loss)
            
            # Detect malicious clients (if clustering enabled)
            if self.use_clustering:
                malicious_clients = self.detect_malicious_clients(client_updates)
                
                # Remove malicious clients from updates
                for client_id in malicious_clients:
                    if client_id in client_updates:
                        del client_updates[client_id]
                
                # Update reputation scores
                if self.use_reputation:
                    self.update_reputation(malicious_clients)
            
            # Server aggregation
            avg_loss = self.server_aggregate(client_updates)
            
            # Evaluate model
            accuracy, f1 = self.evaluate()
            
            # Record results
            self.results['server']['accuracy'].append(accuracy)
            self.results['server']['train_loss'].append(avg_loss)
            self.results['server']['f1_score'].append(f1)
            self.results['server']['phase'].append(1)
            
            # Print progress
            print(f"Round {round_idx+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            
            # Save intermediate results every 10 rounds
            if (round_idx + 1) % 10 == 0 or round_idx == self.args.phase1_rounds - 1:
                self.save_results()
        
        # Transition to Phase 2
        self.current_phase = 2
        print("\nPhase 2 - Classification Layer Training:")
        
        # For each round in phase 2
        for round_idx in range(self.args.phase1_rounds, self.args.total_rounds):
            self.current_round = round_idx
            
            # Select clients for this round
            selected_clients = self.select_clients(self.args.phase2_client_ratio)
            
            # Print round info
            print(f"\nRound {round_idx+1}/{self.args.total_rounds} (Phase 2): {len(selected_clients)} clients selected")
            
            # Client update
            client_updates = {}
            for client_id in tqdm(selected_clients, desc="Client Training"):
                # Skip if client in blacklist
                if hasattr(self, 'blacklisted_clients') and client_id in self.blacklisted_clients:
                    continue
                    
                # Perform client update
                update, loss = self.client_update(client_id, phase=2)
                client_updates[client_id] = (update, loss)
            
            # Server aggregation
            avg_loss = self.server_aggregate(client_updates)
            
            # Evaluate model
            accuracy, f1 = self.evaluate()
            
            # Record results
            self.results['server']['accuracy'].append(accuracy)
            self.results['server']['train_loss'].append(avg_loss)
            self.results['server']['f1_score'].append(f1)
            self.results['server']['phase'].append(2)
            
            # Print progress
            print(f"Round {round_idx+1} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, F1 Score: {f1:.4f}")
            
            # Save intermediate results every 10 rounds
            if (round_idx + 1) % 10 == 0 or round_idx == self.args.total_rounds - 1:
                self.save_results()
        
        # Save final results
        result_file = self.save_results()
        print(f"\nFedCVG training completed")
        print(f"Final results saved to: {result_file}")
        
        return self.results

def main():
    """Main function to run FedCVG algorithm"""
    args = get_fedhyb_args()
    fedcvg = FedCVG(args)
    fedcvg.train()

if __name__ == "__main__":
    main() 