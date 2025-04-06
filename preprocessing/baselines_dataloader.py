import torch
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import Subset, DataLoader
import os
from PIL import Image
import numpy as np
import random


def load_data(name, root='./data', download=True, save_pre_data=True):
    """Load dataset using PIL for image processing"""
    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST',  "IMAGENET", 'CIFAR100']
    assert name in data_dict, "Dataset not supported"

    if not os.path.exists(root):
        os.makedirs(root, exist_ok=True)

    if name == 'MNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        trainset = torchvision.datasets.MNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.MNIST(root=root, train=False, download=download, transform=transform)

    elif name == 'EMNIST':
        # byclass, bymerge, balanced, letters, digits, mnist
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.EMNIST(root=root, train=True, split= 'letters', download=download, transform=transform)
        testset = torchvision.datasets.EMNIST(root=root, train=False, split= 'letters', download=download, transform=transform)

    elif name == 'FashionMNIST':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,))
        ])
        trainset = torchvision.datasets.FashionMNIST(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.FashionMNIST(root=root, train=False, download=download, transform=transform)

    elif name == 'CelebA':
        # Could not loaded possibly for google drive break downs, try again at week days
        target_transform = transforms.Compose([transforms.ToTensor()])
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        trainset = torchvision.datasets.CelebA(root=root, split='train', target_type=list, download=download, transform=transform, target_transform=target_transform)
        testset = torchvision.datasets.CelebA(root=root, split='test', target_type=list, download=download, transform=transform, target_transform=target_transform)

    elif name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
        ])
        trainset = torchvision.datasets.CIFAR10(root=root, train=True, download=download, transform=transform)
        testset = torchvision.datasets.CIFAR10(root=root, train=False, download=download, transform=transform)
        trainset.targets = torch.tensor(trainset.targets, dtype=torch.long)
        testset.targets = torch.tensor(testset.targets, dtype=torch.long)

    elif name == 'CIFAR100':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        trainset = torchvision.datasets.CIFAR100(root=root, train=True, transform=transform, download=True)
        testset = torchvision.datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
        trainset.targets = torch.tensor(trainset.targets, dtype=torch.long)
        testset.targets = torch.tensor(testset.targets, dtype=torch.long)

    elif name == 'QMNIST':
        transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize((0.1307,), (0.3081,))])
        trainset = torchvision.datasets.QMNIST(root=root, what='train', compat=True, download=download, transform=transform)
        testset = torchvision.datasets.QMNIST(root=root, what='test', compat=True, download=download, transform=transform)

    elif name == 'SVHN':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        trainset = torchvision.datasets.SVHN(root=root, split='train', download=download, transform=transform)
        testset = torchvision.datasets.SVHN(root=root, split='test', download=download, transform=transform)
        trainset.targets = torch.tensor(trainset.labels, dtype=torch.long)
        testset.targets = torch.tensor(testset.labels, dtype=torch.long)

    elif name == 'IMAGENET':
        train_val_transform = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20, resample=PIL.Image.BILINEAR),
            transforms.ToTensor(),
        ])
        test_transform = transforms.Compose([
            transforms.ColorJitter(hue=.05, saturation=.05),
            transforms.ToTensor(),
        ])
        # transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
        #                          std=[0.229, 0.224, 0.225])])
        trainset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/train', transform=train_val_transform)
        testset = torchvision.datasets.ImageFolder(root='./data/tiny-imagenet-200/val', transform=test_transform)
        trainset.targets = torch.Tensor(trainset.targets)
        testset.targets = torch.Tensor(testset.targets)

    len_classes_dict = {
        'MNIST': 10,
        'EMNIST': 26, # ByClass: 62. ByMerge: 814,255 47.Digits: 280,000 10.Letters: 145,600 26.MNIST: 70,000 10.
        'FashionMNIST': 10,
        'CelebA': 0,
        'CIFAR10': 10,
        'QMNIST': 10,
        'SVHN': 10,
        'IMAGENET': 200,
        'CIFAR100': 100
    }

    len_classes = len_classes_dict[name]
    
    return trainset, testset, len_classes


def divide_data_iid(trainset, num_client, num_classes, i_seed=0):
    """
    Independent and Identically Distributed (IID) data partition
    Each client receives randomly and uniformly distributed data
    """
    torch.manual_seed(i_seed)
    random.seed(i_seed)
    np.random.seed(i_seed)
    
    trainset_config = {
        'users': [],
        'user_data': {},
        'num_samples': []
    }
    
    # Assign ID to each client
    for i in range(num_client):
        trainset_config['users'].append(f'client_{i:05d}')
    
    # Randomly shuffle data indices
    indices = torch.randperm(len(trainset)).tolist()
    
    # Calculate data volume that should be allocated to each client
    samples_per_client = len(indices) // num_client
    
    # Allocate data to each client
    for i, client_id in enumerate(trainset_config['users']):
        start_idx = i * samples_per_client
        end_idx = (i + 1) * samples_per_client if i < num_client - 1 else len(indices)
        client_indices = indices[start_idx:end_idx]
        
        trainset_config['user_data'][client_id] = Subset(trainset, client_indices)
        trainset_config['num_samples'].append(len(client_indices))
    
    return trainset_config

def divide_data_non_iid_label_skew(trainset, num_client, i_seed=0, alpha=0.5):
    """
    Non-Independent and Identically Distributed (Non-IID) data partition - Label Skew
    Using Dirichlet distribution to control label skew
    
    Parameters:
    - trainset: Training dataset
    - num_client: Number of clients
    - i_seed: Random seed
    - alpha: Parameter of Dirichlet distribution, controls the degree of label distribution imbalance
            Smaller alpha means more imbalanced label distribution; larger alpha means more balanced
    """
    torch.manual_seed(i_seed)
    random.seed(i_seed)
    np.random.seed(i_seed)
    
    trainset_config = {
        'users': [],
        'user_data': {},
        'num_samples': []
    }
    
    # Assign ID to each client
    for i in range(num_client):
        trainset_config['users'].append(f'client_{i:05d}')
    
    # Get all classes and labels in the dataset
    targets = trainset.targets
    if isinstance(targets, torch.Tensor):
        targets = targets.numpy()
    elif not isinstance(targets, np.ndarray):
        targets = np.array(targets)
    
    num_classes = len(np.unique(targets))
    
    # Group data indices by class
    class_indices = {}
    for cls in range(num_classes):
        class_indices[cls] = np.where(targets == cls)[0]
    
    # Generate class distribution for each client using Dirichlet distribution
    client_label_distributions = np.random.dirichlet(np.repeat(alpha, num_classes), size=num_client)
    
    # Calculate total data volume that should be allocated to each client (as balanced as possible)
    total_samples = len(trainset)
    target_samples_per_client = total_samples // num_client
    
    # Allocate data to each client
    for i, client_id in enumerate(trainset_config['users']):
        client_indices = []
        
        # Class distribution for this client
        label_distribution = client_label_distributions[i]
        
        # Calculate number of samples that should be allocated for each class
        target_samples_per_class = (label_distribution * target_samples_per_client).astype(int)
        
        # Ensure total samples across all classes equals the target samples
        remaining_samples = target_samples_per_client - target_samples_per_class.sum()
        if remaining_samples > 0:
            # Allocate remaining samples to classes with highest probability
            sorted_classes = np.argsort(-label_distribution)
            for j in range(remaining_samples):
                cls = sorted_classes[j % len(sorted_classes)]
                target_samples_per_class[cls] += 1
        
        # Allocate samples for each class
        for cls in range(num_classes):
            if target_samples_per_class[cls] > 0:
                # Ensure not exceeding available samples for this class
                num_samples = min(target_samples_per_class[cls], len(class_indices[cls]))
                
                if num_samples > 0:
                    # Randomly select data for this class
                    selected_indices = np.random.choice(class_indices[cls], num_samples, replace=False)
                    client_indices.extend(selected_indices)
                    # Update remaining class indices to avoid duplicate allocation
                    class_indices[cls] = np.setdiff1d(class_indices[cls], selected_indices)
        
        trainset_config['user_data'][client_id] = Subset(trainset, client_indices)
        trainset_config['num_samples'].append(len(client_indices))
    
    return trainset_config

def divide_data(num_client=10, dataset_name='MNIST', i_seed=0, 
                distribution_type='non_iid_label', alpha=0.5):
    """
    Main function for dataset partition
    
    Parameters:
    - num_client: Number of clients
    - dataset_name: Dataset name
    - i_seed: Random seed
    - distribution_type: Data distribution type
                        'iid': Independent and Identically Distributed
                        'non_iid_label': Non-IID Label Skew (using Dirichlet distribution to control label skew)
    - alpha: Dirichlet distribution parameter, controls label skew degree, smaller means more unbalanced
    """
    torch.manual_seed(i_seed)
    random.seed(i_seed)
    np.random.seed(i_seed)

    trainset, testset, num_classes = load_data(dataset_name, download=True)
    
    # Choose different partition methods based on distribution type
    if distribution_type == 'iid':
        trainset_config = divide_data_iid(trainset, num_client, num_classes, i_seed)
    elif distribution_type == 'non_iid_label':
        # Control label skew using Dirichlet distribution with alpha parameter
        trainset_config = divide_data_non_iid_label_skew(trainset, num_client, i_seed, alpha)
    else:
        raise ValueError(f"Unsupported data distribution type: {distribution_type}")
    
    return trainset_config, testset


if __name__ == "__main__":
    # 'MNIST', 'EMNIST', 'FashionMNIST', 'CelebA', 'CIFAR10', 'QMNIST', 'SVHN'
    data_dict = ['MNIST', 'EMNIST', 'FashionMNIST', 'CIFAR10', 'QMNIST', 'SVHN']

    for name in data_dict:
        print(name)
        divide_data(num_client=20, dataset_name=name, i_seed=0)