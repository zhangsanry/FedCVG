# -*- coding: utf-8 -*-

"""
FedCVG algorithm main entry
"""

import os
import torch
import random
import numpy as np
from fedcvg import FedCVG
from federated_parser import get_fedhyb_args

def set_seed(seed):
    """Set random seeds to ensure experiment reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)
    # Set OMP_NUM_THREADS=1 to avoid memory leaks with KMeans on Windows
    os.environ['OMP_NUM_THREADS'] = '1'

def main():
    """Main function"""
    # Get command line arguments
    args = get_fedhyb_args()
    
    # Set random seed
    set_seed(args.i_seed)
    
    # Print experiment configuration
    print("\n" + "="*50)
    print("FedCVG Experiment Configuration:")
    print(f"Dataset: {args.dataset}")
    print(f"Model: {args.model}")
    print(f"Data distribution type: {args.distribution_type}")
    print(f"Number of clients: {args.num_client}")
    print(f"Phase 1 rounds: {args.phase1_rounds}")
    print(f"Phase 2 rounds: {args.phase2_rounds}")
    print(f"Total rounds: {args.total_rounds}")
    print(f"Phase 1 client ratio: {args.phase1_client_ratio}")
    print(f"Phase 2 client ratio: {args.phase2_client_ratio}")
    print(f"Phase 2 learning rate: {args.phase2_learning_rate}")
    print(f"Phase 1 algorithm: {'SCAFFOLD' if args.scaffold_for_phase1 else 'FedProx'}")
    print("="*50 + "\n")
    
    # Create results directory
    if not os.path.exists(args.res_root):
        os.makedirs(args.res_root)
    
    # Initialize and run FedCVG
    fedcvg = FedCVG(args)
    results = fedcvg.train()
    
    print("\n" + "="*50)
    print("FedCVG training completed!")
    print(f"Results saved to: {os.path.abspath(args.res_root)}")
    
    # List files in results directory
    try:
        result_files = os.listdir(args.res_root)
        fedcvg_files = [f for f in result_files if f.startswith('[FedCVG_')]
        
        if fedcvg_files:
            print(f"\nLatest FedCVG result files:")
            # Sort by modification time, display the most recent files
            sorted_files = sorted(fedcvg_files, 
                                  key=lambda x: os.path.getmtime(os.path.join(args.res_root, x)),
                                  reverse=True)
            
            for i, file in enumerate(sorted_files[:3]):  # Only show the 3 most recent files
                file_path = os.path.join(args.res_root, file)
                file_time = os.path.getmtime(file_path)
                file_size = os.path.getsize(file_path)
                import time
                time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(file_time))
                print(f"{i+1}. {file} ({file_size/1024:.1f} KB, {time_str})")
                
            if len(sorted_files) > 3:
                print(f"...and {len(sorted_files)-3} other FedCVG result files")
        else:
            print("Warning: No FedCVG result files found!")
    except Exception as e:
        print(f"Error listing result files: {str(e)}")
    
    print("="*50 + "\n")

if __name__ == "__main__":
    main() 