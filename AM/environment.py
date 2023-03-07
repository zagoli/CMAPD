import torch
import random
from parameters import params
from function import characteristic_function, parallel_pbs, predict_costs

GRID = params['environment']['map']
MAX_COLLECTIVE_SIZE = params['environment']['max_collective_size']


'''
CAPACITY = params['environment']['capacity']

distance_matrix = np.load('./env/distance_matrix.npy')

def best_insertion(waypoints, task):
    waypoints = waypoints[:-2]
    num_waypoints = (waypoints != -1).all(dim=-1).sum(dim=-1)
    pickup = F.pad(task[:2].unsqueeze(0), (0, 1, 0, 0), value=1)
    delivery = F.pad(task[2:].unsqueeze(0), (0, 1, 0, 0), value=-1)

    best = float("inf")
    for q1 in range(1, 1 + num_waypoints):
        for q2 in range(q1, 1 + num_waypoints):
            insertion = torch.cat([waypoints[:q1], pickup, waypoints[q1:q2], delivery, waypoints[q2:]])
            if (insertion[..., 2].cumsum(dim=-1) > CAPACITY).any():
                continue

            dist = 0
            for fp, tp in zip(insertion[:-1], insertion[1:]):
                if (tp == -1).all(dim=-1):
                    break
                dist += distance_matrix[fp[0]][fp[1]][tp[0]][tp[1]]

            if dist < best:
                best = dist
                best_insertion = insertion

    return best_insertion
'''


class Collective:
    def __init__(self, agents, tasks, assignments=None, paths=None, waypoints=None):
        self._batch_size, self._num_agents, _ = agents.size()
        self._device = agents.device

        self.agents = agents
        self.tasks = tasks

        self.indices = torch.empty(
            self._batch_size, self._num_agents, MAX_COLLECTIVE_SIZE,
            dtype=torch.long, device=self._device
        ).fill_(-1) if assignments is None else assignments

        self.paths = torch.cat([agents, agents], dim=-1) if paths is None else paths

        self.waypoints = torch.cat([
            agents.unsqueeze(2),
            -torch.ones_like(agents).unsqueeze(2).repeat(1, 1, 2 * MAX_COLLECTIVE_SIZE, 1)
        ], dim=2) if waypoints is None else waypoints

        ''' code for best_insertion
        self.waypoints = torch.cat([
            F.pad(agents.unsqueeze(2), (0, 1, 0, 0), value=0),
            F.pad(-torch.ones_like(agents).unsqueeze(2).repeat(1, 1, 2 * MAX_COLLECTIVE_SIZE, 1), (0, 1, 0, 0), value=-1)
        ], dim=2) if waypoints is None else waypoints
        '''

        self.is_terminal = torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)

    def _get_paths(self, waypoints, a_idx):
        last_idx = (waypoints[range(self._batch_size), a_idx] != -1).all(dim=-1).sum(dim=-1)
        last_idx = torch.where(last_idx == waypoints.size(2), last_idx - 2, last_idx)

        start = self.waypoints[range(self._batch_size), a_idx, 0]
        end = self.waypoints[range(self._batch_size), a_idx, last_idx]

        ''' code for best_insertion
        last_idx = (waypoints[range(self._batch_size), a_idx] != -1).all(dim=-1).sum(dim=-1)

        start = self.waypoints[range(self._batch_size), a_idx, 0, :-1]
        end = self.waypoints[range(self._batch_size), a_idx, last_idx, :-1]
        '''

        return torch.cat([start, end], dim=-1)

    def add_participant(self, action):
        a_idx, t_idx = action

        insert_idx = (self.indices[range(self._batch_size), a_idx] != -1).sum(dim=-1)
        insert_idx = torch.where((insert_idx < MAX_COLLECTIVE_SIZE - 1) & (~self.is_terminal),
                                 insert_idx, torch.empty_like(insert_idx).fill_(-1))

        collective = Collective(self.agents, self.tasks)

        collective.indices = self.indices.clone()
        collective.indices[range(self._batch_size), a_idx, insert_idx] = torch.where(
            self.is_terminal, collective.indices[..., -1].gather(-1, a_idx.unsqueeze(-1)).squeeze(), t_idx)

        collective.waypoints = self.waypoints.clone()
        indices = (collective.waypoints != -1).all(dim=-1).sum(dim=-1)
        indices = torch.where(indices == collective.waypoints.size(2), indices - 2, indices)

        collective.waypoints[range(self._batch_size), a_idx, indices[range(self._batch_size), a_idx]] = torch.where(
            self.is_terminal.unsqueeze(-1),
            collective.waypoints[range(self._batch_size), a_idx, -1],
            self.tasks[range(self._batch_size), t_idx, :2])
        collective.waypoints[range(self._batch_size), a_idx, 1 + indices[range(self._batch_size), a_idx]] = torch.where(
            self.is_terminal.unsqueeze(-1),
            collective.waypoints[range(self._batch_size), a_idx, -1],
            self.tasks[range(self._batch_size), t_idx, 2:])

        ''' Code for best_insertion
        collective.waypoints[range(self._batch_size), a_idx] = torch.stack([
            best_insertion(self.waypoints[i, a_idx[i]], self.tasks[i, t_idx[i]])
        for i in range(self._batch_size)])
        '''

        collective.paths = self.paths.clone()
        collective.paths[range(self._batch_size), a_idx] = self._get_paths(collective.waypoints, a_idx)

        collective.is_terminal = (collective.indices != -1).sum(dim=-1).sum(dim=-1) == (self.tasks != -1).all(
            dim=-1).sum(dim=-1)

        return collective

    # Adrià previous sequential version
    # def get_reward(self):
    #     reward = torch.zeros(self._batch_size, dtype=torch.float, device=self._device)
    #     for b, terminal in enumerate(self.is_terminal):
    #         if terminal:
    #             reward[b] = characteristic_function(self.waypoints[b].cpu())
    #             ''' Code for best_insertion
    #             reward[b] = func(self.waypoints[b, ..., :-1].cpu())
    #             '''
    #     return reward

    def get_reward(self):
        terminal_indexes = [index for index, terminal in enumerate(self.is_terminal) if terminal]
        waypoints = [self.waypoints[index] for index in terminal_indexes]
        waypoints = waypoints_tensors_to_lists(waypoints)
        costs = parallel_pbs(waypoints)
        reward = torch.zeros(self._batch_size, dtype=torch.float, device=self._device)
        for i, cost in enumerate(costs):
            index = terminal_indexes[i]
            reward[index] = cost
        return reward


def waypoints_tensors_to_lists(waypoints):
    # from list of tensor to list of lists of points (list of int, int)
    result = []
    for w in waypoints:
        w = [[point for point in points if point[0] != -1] for points in w]
        w = [[[int_value.item() for int_value in tensor] for tensor in sublist] for sublist in w]  # ChatGPT
        result.append(w)
    return result


with open(GRID, 'r') as f:
    f.readline()
    grid = [l.strip() for l in f.readlines()]


def sample_agents_tasks(n_agents, n_tasks):
    typecell = {'.': [], 'e': [], '@': []}
    for i, row in enumerate(grid):
        for j, cell in enumerate(row):
            typecell[cell].append((i, j))
    random.shuffle(typecell['e'])
    return ([typecell['e'].pop() for _ in range(n_agents)],
            [[typecell['e'].pop(), typecell['e'].pop()] for _ in range(n_tasks)])
