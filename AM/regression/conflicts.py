# Jacopo Zagoli, 06/02/2023
from dataclasses import dataclass


@dataclass
class Conflict:
    first_agent: int
    second_agent: int
    timestep: int
    first_position: list
    second_position: list
    conflict_type: str


class ConflictsFinder:
    def __init__(self, paths: list):
        self.__paths = paths

    def get_conflicts(self)-> list[Conflict]:
        conflicts = []
        n_paths = len(self.__paths)
        for i in range(n_paths):
            for j in range(i + 1, n_paths):
                conflicts += self.__detect_conflicts(i, j)
        return conflicts

    def __detect_conflicts(self, first_agent, second_agent)->list[Conflict]:
        conflicts = []
        length = max(len(self.__paths[first_agent]), len(self.__paths[second_agent]))
        for timestep in range(length):
            first_pos = self.__get_agent_pos(first_agent, timestep)
            second_pos = self.__get_agent_pos(second_agent, timestep)
            # Vertex
            if first_pos == second_pos:
                conflicts.append(
                    Conflict(first_agent, second_agent, timestep, first_pos, second_pos, 'VERTEX')
                )
            # Edge
            if timestep < length - 1:
                first_next_pos = self.__get_agent_pos(first_agent, timestep + 1)
                second_next_pos = self.__get_agent_pos(second_agent, timestep + 1)
                if first_pos == second_next_pos and second_pos == first_next_pos:
                    conflicts.append(
                        Conflict(first_agent, second_agent, timestep + 1, first_pos, first_next_pos, 'EDGE')
                    )
        return conflicts

    def __get_agent_pos(self, agent: int, timestep: int):
        assert timestep >= 0
        path = self.__paths[agent]
        if timestep < len(path):
            return path[timestep]
        return path[-1]
