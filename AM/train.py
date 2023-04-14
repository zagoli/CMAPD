import torch
import random
from model import Transformer
from trainer import Trainer
from parameters import params

# Initialize the seed for random operations
SEED = params['setup']['seed']

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Initialize CUDA if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print('Using {}'.format(device))

# Initialize model and baseline
model = Transformer(
    input_size=params['model']['input_size'],
    d_model=params['model']['d_model'],
    nhead=params['model']['nhead'],
    dim_feedforward=params['model']['dim_feedforward'],
    num_layers=params['model']['num_layers'])
baseline = Transformer(
    input_size=params['model']['input_size'],
    d_model=params['model']['d_model'],
    nhead=params['model']['nhead'],
    dim_feedforward=params['model']['dim_feedforward'],
    num_layers=params['model']['num_layers'])

# Initialize the trainer
trainer = Trainer(
    model=model,
    baseline=baseline,
    learning_rate=params['training']['learning_rate'],
    batch_size=params['training']['batch_size'],
    device=device)

# Train the model
trainer.train(
    n_agents=params['training']['n_agents'],
    n_tasks=params['training']['n_tasks'],
    train_size=params['training']['train_size'],
    eval_size=params['training']['eval_size'],
    n_epochs=params['training']['n_epochs'],
    output_name='CONVERGENCE_TEST',
    log_prefix='CONVERGENCE_TEST'
)
