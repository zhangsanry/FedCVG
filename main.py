from federated_parser import get_args
from federated_learning import FederatedLearning
import os

def main():
    """
    Federated Learning main program
    Running example:
    python main.py --dataset MNIST --model LeNet --num_client 10 --num_local_class 2 --num_round 100
    """
    # Set OMP_NUM_THREADS=1 to avoid memory leaks with KMeans on Windows
    os.environ['OMP_NUM_THREADS'] = '1'
    
    # Get parameter configuration
    args = get_args()
    
    # Create and run federated learning instance
    fed_learning = FederatedLearning(args)
    fed_learning.train()

if __name__ == "__main__":
    main() 