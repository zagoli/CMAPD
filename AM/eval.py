import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from environment import Collective

import torch
import random
from model import Transformer
from parameters import params


# Initialize the seed for random operations
SEED = params['setup']['seed']

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Initialize CUDA if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"  # temp solution

def _sample_action(probs, deterministic=False):
    tasks_size = probs.size(2)
    p = probs.flatten(-2)
    ij = p.argmax(-1) if deterministic else torch.multinomial((p == 0).all(dim=-1, keepdim=True) + p, 1).squeeze()
    action = (ij // tasks_size, ij % tasks_size)
    logprob = torch.log(p[range(probs.size(0)), ij])
    return action, logprob

def _rollout(tasks, collective, policy, stochastic=False):
    policy.eval()
    while not collective.is_terminal.all():
        probs = policy(tasks, collective)
        if stochastic:
            action, _ = _sample_action(probs)
        else:
            action, _ = _sample_action(probs, deterministic=True)
        collective = collective.add_participant(action)
    reward = collective.get_reward()
    return reward

def _random_rollout(collective):
    n_tasks = 20
    n_agents = 10
    task_indexes = list(range(n_tasks))
    while not collective.is_terminal.all():
        t_idx = random.choice(task_indexes)
        task_indexes.remove(t_idx)
        a_idx = random.randrange(0,n_agents)
        action = (torch.tensor([a_idx]), torch.tensor([t_idx]))
        collective = collective.add_participant(action)
    reward = collective.get_reward()
    return reward


if __name__ == '__main__':
    # Initialize model
    model = Transformer(
        input_size=params['model']['input_size'],
        d_model=params['model']['d_model'],
        nhead=params['model']['nhead'],
        dim_feedforward=params['model']['dim_feedforward'],
        num_layers=params['model']['num_layers'])
    model.load_state_dict(torch.load('models/plain_old_pbs.pth'))

    for i in range(100):
        # Read agents and tasks
        with open(f'instances/maps/{i}.map', 'r') as f:
            dx, dy = map(int, f.readline().strip().split(','))
            for _ in range(3): f.readline()
            typecell = {'.': [], 'e': [], 'r': [], '@': []}
            for i, row in enumerate(f.readlines()):
                for j, cell in enumerate(row.strip()):
                    typecell[cell].append((i, j))
            agents = torch.tensor([typecell['r']], device=device)

        with open(f'instances/tasks/{i}.task', 'r') as f:
            f.readline()
            tasks = []
            for row in f.readlines():
                tasks.append(list(map(int, row.strip().split('\t')[1:3])))
            
            tasks = torch.tensor([tasks], device=device)
            tasks = torch.stack([tasks[..., 0] // dy, tasks[..., 0] % dy, tasks[..., 1] // dy, tasks[..., 1] % dy], dim=-1)


        collective = Collective(agents, tasks)
        reward = _rollout(tasks, collective, model)
        print(f"model result:{reward.item()}")
