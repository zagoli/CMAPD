params = {
    'setup': {
        'seed': 42                      # Random seed for random and torch packages
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
        'dim_feedforward': 512,         # Size of the hidden dimensions inside the feedforwards layers
        'num_layers': 3                 # Number of encoder blocks
    },
    'training': {
        'n_agents': 10,                 # Number of agents in the training instances
        'n_tasks': [20, 25],            # Range of tasks in the training instances
        'batch_size': 8,#1024,    #4096          # Batch size to train the model
        'train_size': 15,#40960,          # Number of training instances (ideal would be > 100k)
        'eval_size': 2,#10240,             # Number of test instances (ideal would be > 10k)
        'learning_rate': 1e-4,          # Learning rate in gradient descent
        'n_epochs': 50                 # Number of training epochs
    }
}
