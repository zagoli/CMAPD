import random
from math import inf

import torch

from function import parallel_pbs
from parameters import params
from regression.grid_solver import GridSolver
from regression.utils import read_grid

GRID = params['environment']['map']
MAX_COLLECTIVE_SIZE = params['environment']['max_collective_size']
grid_for_solver, _ = read_grid(GRID)
grid_solver = GridSolver(grid_for_solver)


def waypoints_respect_capacity(new_waypoints, pickups):
    cum_capacity = 0
    for w in new_waypoints[1:]:  # exclude agent starting point
        if w in pickups:
            cum_capacity += 1
        else:
            cum_capacity -= 1
        if cum_capacity > params['environment']['capacity']:
            return False
    return True


def insert_pickup(agent_waypoints, pickup, pickups_for_agent):
    # try to insert pickup in every possible position after the first (agent starting point)
    best_waypoints, best_waypoints_cost, pickup_index = None, inf, 0
    for i in range(1, len(agent_waypoints) + 1):
        new_waypoints = agent_waypoints[:i] + [pickup] + agent_waypoints[i:]
        # checks if the waypoints are consistent with capacity constraint
        if waypoints_respect_capacity(new_waypoints, pickups_for_agent):
            waypoints_cost = grid_solver.get_waypoints_path(new_waypoints)
            if waypoints_cost < best_waypoints_cost:
                best_waypoints = new_waypoints
                best_waypoints_cost = waypoints_cost
                pickup_index = i
    return best_waypoints, pickup_index


def insert_delivery(agent_waypoints, delivery, pickup_index):
    # try to insert pickup in every possible position after the pickup
    best_waypoints, best_waypoints_cost = None, inf
    for i in range(pickup_index + 1, len(agent_waypoints) + 1):  # Search after the pickup
        new_waypoints = agent_waypoints[:i] + [delivery] + agent_waypoints[i:]
        waypoints_cost = grid_solver.get_waypoints_path(new_waypoints)
        if waypoints_cost < best_waypoints_cost:
            best_waypoints = new_waypoints
            best_waypoints_cost = waypoints_cost
    return best_waypoints


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
        self.is_terminal = torch.zeros(self._batch_size, dtype=torch.bool, device=self._device)

    def _get_paths(self, waypoints, a_idx):
        last_idx = (waypoints[range(self._batch_size), a_idx] != -1).all(dim=-1).sum(dim=-1)
        last_idx = torch.where(last_idx == waypoints.size(2), last_idx - 2, last_idx)
        # input di torch.where(condition, input, other)

        start = self.waypoints[range(self._batch_size), a_idx, 0]
        end = self.waypoints[range(self._batch_size), a_idx, last_idx]
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
        """
        indices = (collective.waypoints != -1).all(dim=-1).sum(dim=-1)
        indices = torch.where(indices == collective.waypoints.size(2), indices - 2, indices)
        """

        for batch_idx in range(self._batch_size):
            new_waypoints = self.best_insertion(batch_idx, a_idx[batch_idx], t_idx[batch_idx])
            collective.waypoints[batch_idx, a_idx[batch_idx]] = new_waypoints

        """
        # insert pickup
        collective.waypoints[range(self._batch_size), a_idx, indices[range(self._batch_size), a_idx]] = torch.where(
            self.is_terminal.unsqueeze(-1),
            collective.waypoints[range(self._batch_size), a_idx, -1],
            self.tasks[range(self._batch_size), t_idx, :2])
        # insert delivery
        collective.waypoints[range(self._batch_size), a_idx, 1 + indices[range(self._batch_size), a_idx]] = torch.where(
            self.is_terminal.unsqueeze(-1),
            collective.waypoints[range(self._batch_size), a_idx, -1],
            self.tasks[range(self._batch_size), t_idx, 2:])
        """

        collective.paths = self.paths.clone()
        collective.paths[range(self._batch_size), a_idx] = self._get_paths(collective.waypoints, a_idx)

        collective.is_terminal = (collective.indices != -1).sum(dim=-1).sum(dim=-1) == (self.tasks != -1).all(
            dim=-1).sum(dim=-1)

        return collective

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

    def best_insertion(self, batch_idx, a_idx, t_idx):
        # list of dimension (max path size, x y coordinates) for a specific agent
        agent_waypoints = self.waypoints[batch_idx, a_idx].detach().clone().squeeze().tolist()
        agent_waypoints = list(filter(lambda pos: pos != [-1, -1], agent_waypoints))
        task = self.tasks[batch_idx, t_idx].detach().clone().squeeze()
        if task.tolist() == [-1, -1, -1, -1]:
            # I can't insert a non-existent task
            return self.waypoints[batch_idx, a_idx]  # The original waypoints tensor
        else:
            pickup = task[0:2].tolist()
            delivery = task[2:4].tolist()
            # get list of all pickup points for the agent
            tasks_idxes_for_agent = self.indices[batch_idx, a_idx].detach().clone().squeeze().tolist()
            pickups_for_agent = self.pickup_locations_for_agent(batch_idx, tasks_idxes_for_agent) + [pickup]
            # insert pickup and delivery
            agent_waypoints, pickup_index = insert_pickup(agent_waypoints, pickup, pickups_for_agent)
            agent_waypoints = insert_delivery(agent_waypoints, delivery, pickup_index)
            result_tensor = torch.full((2 * MAX_COLLECTIVE_SIZE + 1, 2), -1)
            result_tensor[:len(agent_waypoints)] = torch.tensor(agent_waypoints)
            return result_tensor


    def pickup_locations_for_agent(self, batch_idx, tasks_idxes_for_agent):
        pickups = []
        for t_idx in tasks_idxes_for_agent:
            if t_idx != -1:
                task_pickup = self.tasks[batch_idx, t_idx, 0:2].tolist()
                pickups.append(task_pickup)
        return pickups


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
