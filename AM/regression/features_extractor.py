# Jacopo Zagoli, 31/01/2023
from regression.utils import ravel
from regression.grid_solver import GridSolver
from regression.conflicts import ConflictsFinder, Conflict


class FeaturesExtractor:

    def __init__(self, assignment: list, grid, grid_size: tuple, grid_solver: GridSolver):
        self.__assignment = assignment
        self.__grid = grid
        self.__grid_size = grid_size
        self.__grid_solver = grid_solver
        self.__paths = self.__compute_paths()
        self.__conflicts = self.__compute_conflicts()

    def __compute_paths(self) -> list:
        paths = []
        for waypoints in self.__assignment:
            path = self.__grid_solver.get_waypoints_distance(waypoints)
            paths.append(path)
        return paths

    def __compute_conflicts(self) -> list[Conflict]:
        finder = ConflictsFinder(self.__paths)
        return finder.get_conflicts()

    def get_features(self) -> list[int]:
        features = []
        features += self.__agents_start_goal()
        features += self.__n_waypoints_per_agent()
        features += self.__paths_length()
        features += self.__n_conflicts()
        features += self.__n_conflicted_agents()
        features += self.__n_conflicted_agent_couples()
        return features

    def __agents_start_goal(self) -> list[int]:
        n_cols = self.__grid_size[1]
        locations = []
        for waypoints in self.__assignment:
            locations += [waypoints[0], waypoints[-1]]
        return [ravel(p, n_cols) for p in locations]

    def __n_waypoints_per_agent(self) -> list[int]:
        return [len(waypoints) for waypoints in self.__assignment]

    def __paths_length(self) -> list[int]:
        return [len(path) for path in self.__paths]

    def __n_conflicts(self) -> list[int]:
        return [len(self.__conflicts)]

    def __n_conflicted_agents(self) -> list[int]:
        conflicted_agents = set()
        for c in self.__conflicts:
            conflicted_agents.add(c.first_agent)
            conflicted_agents.add(c.second_agent)
        return [len(conflicted_agents)]

    def __n_conflicted_agent_couples(self) -> list[int]:
        # in conflicts list, conflicts are ordered by increasing first agent
        couples = set()
        for conflict in self.__conflicts:
            couples.add((conflict.first_agent, conflict.second_agent))
        return [len(couples)]