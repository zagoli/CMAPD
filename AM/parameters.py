params = {
    'setup': {
        'seed': 10                      # Random seed for random and torch packages
    },
    'environment': {
        'map': './env/grid.map',        # Filepath of the map file
        'max_collective_size': 5,       # Maximum collective size (hard coded into environment.py)
        'capacity': 3                   # Maximum agent capacity (hard coded into environment.py)
    },
    'model': {                          # Embedding -> N x (Attention -> Feedforward) -> Decoder
        'input_size': [21, 35, 21, 35], # Input feature size (embedding layer pytorch) [pickup_x, pickup_y, delivery_x, delivery_y]
        'd_model': 128,                 # Size of the hidden dimensions inside the attention model
        'nhead': 8,                     # Number of heads inside the multi-head attention mechanism
        'dim_feedforward': 512,         # Size of the hidden dimansionis inside the feedforwards layers
        'num_layers': 3                 # Number of encoder blocks
    },
    'training': {
        'n_agents': 10,                 # Number of agents in the training instances
        'n_tasks': [20, 20],            # Range of tasks in the trainig instances
        'batch_size': 256,              # Batch size to train the model
        'train_size': 2048,             # Number of training instances (ideal would be > 100k)
        'eval_size': 256,               # Number of test instances (ideal would be > 10k)
        'learning_rate': 0.0001,        # Learning rate in gradient descent
        'n_epochs': 1                 # Number of training epochs
    }
}
