import argparse

def get_args():
    """
    Get all parameters required for federated learning, including basic algorithm and FedCVG specific parameters
    """
    parser = argparse.ArgumentParser(description='Federated Learning Parameter Configuration')
    
    # System related parameters
    parser.add_argument('--dataset', type=str, default='CIFAR10', 
                        choices=['MNIST', 'CIFAR10', 'FashionMNIST', 'SVHN', 'CIFAR100'],
                        help='Dataset selection')
    parser.add_argument('--model', type=str, default='ResNet34',
                        choices=["LeNet", 'CNN', 'AlexCifarNet', "ResNet18", "ResNet34", "ResNet50"],
                        help='Model selection')
    parser.add_argument('--num_client', type=int, default=10, help='Number of clients')
    parser.add_argument('--client_ratio', type=float, default=0.5, help='Ratio of clients participating in each round of federated learning, range [0,1]')
    parser.add_argument('--num_round', type=int, default=100, help='Number of federated learning rounds')
    parser.add_argument('--i_seed', type=int, default=1234, help='Random seed')
    parser.add_argument('--res_root', type=str, default='./results', help='Results saving path')
    
    # Data distribution related parameters
    parser.add_argument('--distribution_type', type=str, default='non_iid_label',
                        choices=['iid', 'non_iid_label'],
                        help='Data distribution type: iid (Independent and Identically Distributed), non_iid_label (control label skew using Dirichlet distribution)')
    parser.add_argument('--alpha', type=float, default=0.1, 
                        help='Dirichlet distribution parameter alpha, controls label skew degree, smaller values mean more imbalance')
    
    # General client training parameters
    parser.add_argument('--fed_algo', type=str, default='FedProx',
                        choices=["FedAvg", "SCAFFOLD", "FedProx", "FedNova", "FedCVG", 
                                "Krum", "MultiKrum", "Bulyan", "TrimmedMean", "Median", "Auror"],
                        help='Federated learning algorithm selection')
    parser.add_argument('--num_local_epoch', type=int, default=3, help='Number of local training epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum factor')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='Weight decay coefficient')
    
    # FedProx specific parameters
    parser.add_argument('--fedprox_mu', type=float, default=0.001, 
                        help='FedProx algorithm proximal term coefficient, controls deviation of local model from global model, only effective when fed_algo=FedProx or FedCVG')
    
    # SCAFFOLD specific parameters
    parser.add_argument('--scaffold_c_lr', type=float, default=0.001, 
                        help='SCAFFOLD algorithm control variable learning rate, only effective when fed_algo=SCAFFOLD')
    
    # FedNova specific parameters
    parser.add_argument('--fednova_tau_eff', type=str, default='uniform', 
                        choices=['uniform', 'n_local_epoch'], 
                        help='Calculation method for normalized weights in FedNova, uniform means uniform weights, n_local_epoch means weighted by number of local iterations, only effective when fed_algo=FedNova')
    
    # FedCVG specific parameters - two-phase training
    parser.add_argument('--phase1_rounds', type=int, default=40, 
                        help='Phase 1 training rounds, only effective when fed_algo=FedCVG')
    parser.add_argument('--phase2_rounds', type=int, default=60, 
                        help='Phase 2 training rounds, only effective when fed_algo=FedCVG')
    parser.add_argument('--phase1_client_ratio', type=float, default=0.5, 
                        help='Ratio of clients participating in each round of phase 1, range [0,1], only effective when fed_algo=FedCVG')
    parser.add_argument('--phase2_client_ratio', type=float, default=0.5, 
                        help='Ratio of clients participating in each round of phase 2, range [0,1], only effective when fed_algo=FedCVG')
    parser.add_argument('--phase2_learning_rate', type=float, default=0.01, 
                        help='Learning rate for phase 2, only effective when fed_algo=FedCVG')
    parser.add_argument('--scaffold_for_phase1', type=bool, default=True,
                        help='Use SCAFFOLD algorithm instead of FedProx in FedCVG phase 1, only effective when fed_algo=FedCVG')
    
    # FedCVG clustering and reputation mechanism parameters
    parser.add_argument('--use_clustering', type=bool, default=True,
                        help='Whether to use clustering mechanism to detect malicious clients in FedCVG phase 1, only effective when fed_algo=FedCVG')
    parser.add_argument('--use_reputation', type=bool, default=True,
                        help='Whether to use reputation mechanism to manage client participation in FedCVG phase 1, only effective when fed_algo=FedCVG')
    parser.add_argument('--clustering_threshold', type=float, default=0.25,
                        help='In reputation mechanism, the threshold for the ratio of times marked as malicious to training rounds, exceeding this threshold will be permanently excluded, only effective when use_reputation=True')
    parser.add_argument('--fedhyb_cluster_count', type=int, default=2, 
                        help='Number of clusters in FedCVG clustering algorithm, usually set to 2 (normal cluster and malicious cluster), only effective when use_clustering=True')
    parser.add_argument('--fedhyb_distance_threshold', type=float, default=1, 
                        help='Distance threshold for determining abnormal clusters in FedCVG clustering algorithm, if inter-cluster distance exceeds this threshold, it is considered a malicious cluster, only effective when use_clustering=True')
    parser.add_argument('--fedhyb_size_threshold', type=float, default=0.45, 
                        help='Size threshold for determining abnormal clusters in FedCVG clustering algorithm, if cluster size ratio is less than this threshold, it may be a malicious cluster, only effective when use_clustering=True')
    
    # Krum / MultiKrum / Bulyan specific parameters
    parser.add_argument('--num_malicious_tolerance', type=int, default=2, 
                        help='Maximum number of malicious clients that robust algorithms can tolerate, only effective when fed_algo=Krum, MultiKrum, or Bulyan')
    parser.add_argument('--multikrum_k', type=int, default=1, 
                        help='Number of clients selected by MultiKrum algorithm, or number of clients selected in phase 1 of Bulyan, only effective when fed_algo=MultiKrum or Bulyan')
    parser.add_argument('--use_distances_as_weights', type=bool, default=False, 
                        help='Whether to use distances as weights for weighted aggregation (rather than simply selecting the best clients), only effective when fed_algo=Krum or MultiKrum')
    parser.add_argument('--bulyan_beta', type=float, default=1.0,
                        help='Bulyan trimming parameter, used to determine how many extreme values to remove, only effective when fed_algo=Bulyan')
    
    # TrimmedMean / Median specific parameters
    parser.add_argument('--trimmed_ratio', type=float, default=0.4, 
                        help='Trimming ratio for TrimmedMean algorithm, indicates what proportion of maximum and minimum values to trim from each coordinate, range [0, 0.5), only effective when fed_algo=TrimmedMean')
    parser.add_argument('--trim_k', type=int, default=None, 
                        help='Direct specification of trimming quantity for TrimmedMean algorithm, takes precedence over trimmed_ratio when set, indicates how many values to trim from each end of each coordinate, only effective when fed_algo=TrimmedMean')
    
    # Auror specific parameters
    parser.add_argument('--auror_n_clusters', type=int, default=2, 
                        help='Number of clusters in Auror clustering algorithm, usually set to 2 (normal cluster and malicious cluster), only effective when fed_algo=Auror')
    parser.add_argument('--auror_distance_threshold', type=float, default=0.5, 
                        help='Distance threshold for determining abnormal clusters in Auror, if inter-cluster distance exceeds this threshold, it is considered a malicious cluster, only effective when fed_algo=Auror')
    parser.add_argument('--auror_size_threshold', type=float, default=0.45, 
                        help='Size threshold for determining abnormal clusters in Auror, if cluster size ratio is less than this threshold, it may be a malicious cluster, only effective when fed_algo=Auror')
    
    # Poisoning attack related parameters
    parser.add_argument('--enable_attack', type=bool, default=False,
                        help='Whether to enable gradient poisoning attack')
    parser.add_argument('--num_malicious', type=int, default=2,
                        help='Number of malicious clients, clients from 0 to num_malicious-1 will conduct poisoning attacks')
    parser.add_argument('--attack_type', type=str, default='sign_flip',
                        choices=['gaussian', 'sign_flip', 'targeted'],
                        help='Attack type: gaussian (add Gaussian noise), sign_flip (flip gradient signs), targeted (targeted attack)')
    parser.add_argument('--noise_level', type=float, default=1.0,
                        help='Noise level, controls the intensity of added noise')
    
    # Gradient recording mechanism parameters
    parser.add_argument('--use_historical_gradients', action='store_true', help='Whether to use historical gradient recording mechanism in phase 2')
    parser.add_argument('--gradient_decay', type=float, default=3, help='Historical gradient decay coefficient, range (0,1]')
    parser.add_argument('--gradient_threshold', type=float, default=0.2, help='Threshold for using historical gradients, historical gradient is not used when less than this value')
    
    args = parser.parse_args()
    
    # Calculate FedCVG total training rounds
    if args.fed_algo == 'FedCVG':
        args.total_rounds = args.phase1_rounds + args.phase2_rounds
    else:
        args.total_rounds = args.num_round
    
    return args

def get_fedhyb_args():
    """
    Get FedCVG algorithm command line arguments (backward compatibility)
    
    Returns:
        args: Parsed arguments
    """
    args = get_args()
    # Ensure algorithm is FedCVG
    args.fed_algo = 'FedCVG'
    # Calculate total training rounds
    args.total_rounds = args.phase1_rounds + args.phase2_rounds
    
    # Ensure relevant parameters are set correctly based on phase 1 algorithm
    if hasattr(args, 'scaffold_for_phase1') and args.scaffold_for_phase1:
        # Ensure SCAFFOLD related parameters are set
        if not hasattr(args, 'scaffold_c_lr') or args.scaffold_c_lr is None:
            args.scaffold_c_lr = 0.001  # Use default value
    else:
        # Ensure FedProx related parameters are set
        if not hasattr(args, 'fedprox_mu') or args.fedprox_mu is None:
            args.fedprox_mu = 0.01  # Use default value
    
    # Ensure phase 2 learning rate is set correctly
    if not hasattr(args, 'phase2_learning_rate') or args.phase2_learning_rate is None:
        args.phase2_learning_rate = 0.001  # Use default value
    
    # Ensure clustering and reputation mechanism parameters are set correctly
    if hasattr(args, 'use_clustering') and args.use_clustering:
        # Ensure clustering related parameters are set
        if not hasattr(args, 'fedhyb_cluster_count'):
            args.fedhyb_cluster_count = 2
        if not hasattr(args, 'fedhyb_distance_threshold'):
            args.fedhyb_distance_threshold = 0.5
        if not hasattr(args, 'fedhyb_size_threshold'):
            args.fedhyb_size_threshold = 0.45
    
    if hasattr(args, 'use_reputation') and args.use_reputation:
        # Ensure reputation mechanism parameters are set
        if not hasattr(args, 'clustering_threshold'):
            args.clustering_threshold = 0.1
    
    return args

if __name__ == "__main__":
    args = get_args()
    print(args) 