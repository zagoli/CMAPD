import logging
import random
from timeit import default_timer as timer

from datetime import datetime
from datetime import timedelta
import torch
from torch.utils.data import DataLoader

from dataset import TADataset
from environment import Collective
from pathlib import Path


def _sample_action(probs, deterministic=False):
    tasks_size = probs.size(2)
    p = probs.flatten(-2)
    ij = p.argmax(-1) if deterministic else torch.multinomial((p == 0).all(dim=-1, keepdim=True) + p, 1).squeeze()
    action = (ij // tasks_size, ij % tasks_size)
    logprob = torch.log(p[range(probs.size(0)), ij])
    return action, logprob


def check_output_dir():
    dir = Path('models')
    if not (dir.exists()):
        dir.mkdir()


def logger_setup():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_filename = datetime.now().strftime("%d-%m-%Y_%H:%M:%S") + '.log'
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)

    logger.addHandler(file_handler)
    logger.addHandler(stream_handler)

    logger.info("Training started at " + str(datetime.now()))
    return logger


class Trainer:
    def __init__(self, model, baseline, learning_rate, batch_size, device):
        self.batch_size = batch_size
        self.device = device

        self.model = model.to(device)
        self.baseline = baseline.to(device)
        self.optim = torch.optim.Adam(self.model.parameters(), lr=learning_rate)

    @torch.no_grad()
    def _generate_assignment_data(self, dataset, policy):
        policy.eval()

        for data in DataLoader(dataset, batch_size=self.batch_size, num_workers=4):
            indices, paths, waypoints = [], [], []

            agents = data['agents'].to(self.device)
            tasks = data['tasks'].to(self.device)
            collective = Collective(agents, tasks)

            while not collective.is_terminal.all():
                indices.append(collective.indices.cpu())
                paths.append(collective.paths.cpu())
                waypoints.append(collective.waypoints.cpu())
                probs = policy(tasks, collective)
                action, _ = _sample_action(probs)
                collective = collective.add_participant(action)

            indices, paths, waypoints = tuple(map(torch.stack, (indices, paths, waypoints)))

            num_states = (tasks != -1).all(dim=-1).sum(dim=-1)
            idx = torch.tensor(list(map(random.randrange, num_states)))

            dataset.assignments.extend(
                indices.gather(0, idx.view(1, -1, 1, 1).expand([-1, -1, indices.size(2), indices.size(3)])).squeeze())
            dataset.paths.extend(
                paths.gather(0, idx.view(1, -1, 1, 1).expand([-1, -1, paths.size(2), paths.size(3)])).squeeze())
            dataset.waypoints.extend(
                waypoints.gather(0, idx.view(1, -1, 1, 1, 1).expand([-1, -1, *waypoints.size()[2:]])).squeeze())

    @torch.no_grad()
    def _rollout(self, tasks, collective, policy, stochastic=False):
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

    def _evaluation(self, dataset, model, baseline):
        model.eval()  # sets module in eval mode
        baseline.eval()

        model_reward = []
        for data in DataLoader(dataset, batch_size=self.batch_size, num_workers=4):
            agents = data['agents'].to(self.device)
            tasks = data['tasks'].to(self.device)
            collective = Collective(agents, tasks)
            reward = self._rollout(tasks, collective, model)
            model_reward.extend(reward.tolist())

        return sum(model_reward) / len(model_reward)

    def _optimize(self, dataset, model):
        model.train()  # sets module in training mode

        self._generate_assignment_data(dataset, self.baseline)
        for data in DataLoader(dataset, batch_size=self.batch_size, num_workers=4, shuffle=True):
            self.optim.zero_grad()

            agents = data['agents'].to(self.device)
            tasks = data['tasks'].to(self.device)
            assignments = data['assignments'].to(self.device)
            paths = data['paths'].to(self.device)
            waypoints = data['waypoints'].to(self.device)

            collective = Collective(agents, tasks, assignments, paths, waypoints)

            probs = model(tasks, collective)
            action, logprob = _sample_action(probs, deterministic=True)
            next_collective = collective.add_participant(action)
            model_reward = self._rollout(tasks, next_collective, model, stochastic=True)

            with torch.no_grad():
                probs = self.baseline(tasks, collective)
                action, _ = _sample_action(probs, deterministic=True)
                next_collective = collective.add_participant(action)
                baseline_reward = self._rollout(tasks, next_collective, self.baseline, stochastic=True)

            advantage = model_reward - baseline_reward
            loss = (advantage * logprob).mean()

            loss.backward()
            self.optim.step()

            for bp, mp in zip(self.baseline.parameters(), self.model.parameters()):
                bp.data.copy_(0.01 * mp.data + (1 - 0.01) * bp.data)

    def train(self, n_agents, n_tasks, train_size, eval_size, n_epochs):
        best = float("inf")
        durations = []
        check_output_dir()
        logger = logger_setup()
        for epoch in range(n_epochs):
            start = timer()
            dataset_train = TADataset(train_size, n_agents, n_tasks)
            dataset_eval = TADataset(eval_size, n_agents, n_tasks)
            self._optimize(dataset_train, self.model)
            model_reward = self._evaluation(
                dataset_eval, self.model, self.baseline)
            if model_reward < best:
                best = model_reward
                torch.save(self.model.state_dict(), Path('models/transformer.pth'))
            stop = timer()
            duration = stop - start
            durations.append(duration)
            eta = (sum(durations) / len(durations)) * (n_epochs - (epoch + 1))
            logger.info(
                f'epoch {epoch:5}, time={timedelta(seconds=duration)}, eta={timedelta(seconds=eta)}, reward={model_reward:5}'
            )
        logger.info("End of training. Best reward: " + str(best))
