# Jacopo Zagoli, 03/02/2023
from collections import deque
import copy


class BfsNode:
    def __init__(self, row, col, cost, path):
        self.row = row
        self.col = col
        self.cost = cost
        self.path = path


class GridSolver:
    def __init__(self, grid):
        self.__grid = grid != b'@'
        self.__path_table = dict()
        self.__compute_paths()

    def __is_present_bidirectional(self, from_point: tuple, to_point: tuple):
        forward = self.__is_present(from_point, to_point)
        reverse = self.__is_present(to_point, from_point)
        return forward or reverse

    def __is_present(self, from_point, to_point):
        forward = from_point in self.__path_table and to_point in self.__path_table[from_point]
        return forward

    def __compute_paths(self):
        n_rows, n_cols = self.__grid.shape
        for row in range(n_rows):
            for col in range(n_cols):
                if self.__grid[row, col]:
                    self.__bfs(row, col)

    def __bfs(self, row, col):
        root = BfsNode(row, col, 0, [[row, col]])
        frontier = deque([root])
        explored = set()
        while frontier:
            vertex = frontier.popleft()
            self.__update_path_table(root, vertex)
            for neighbour in self.__vertex_neighbours(vertex):
                neighbour_pos = (neighbour.row, neighbour.col)
                if neighbour_pos not in explored:
                    explored.add(neighbour_pos)
                    frontier.append(neighbour)

    def __vertex_neighbours(self, vertex: BfsNode):
        neighbours = []
        moves = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        for r_move, c_move in moves:
            new_row = vertex.row + r_move
            new_col = vertex.col + c_move
            if self.__is_in_grid(new_row, new_col) and self.__grid[new_row, new_col]:
                new_path = copy.copy(vertex.path)
                new_path.append([new_row, new_col])
                neighbour = BfsNode(new_row, new_col, vertex.cost + 1, new_path)
                neighbours.append(neighbour)
        return neighbours

    def __is_in_grid(self, row, col):
        n_rows, n_cols = self.__grid.shape
        return 0 <= row < n_rows and 0 <= col < n_cols

    def __update_path_table(self, root, vertex):
        from_point = (root.row, root.col)
        to_point = (vertex.row, vertex.col)
        if from_point != to_point and not self.__is_present_bidirectional(from_point, to_point):
            self.__init_dict(from_point)
            self.__path_table[from_point][to_point] = vertex.path

    def __init_dict(self, point):
        if point not in self.__path_table:
            self.__path_table[point] = dict()

    def get_path(self, from_point: list, to_point: list):
        if from_point == to_point:
            return from_point
        from_point_index = (from_point[0], from_point[1])
        to_point_index = (to_point[0], to_point[1])
        if self.__is_present(from_point_index, to_point_index):
            return self.__path_table[from_point_index][to_point_index]
        if self.__is_present(to_point_index, from_point_index):
            return self.__path_table[to_point_index][from_point_index][::-1]
        raise Exception(f'No path is present from {from_point} to {to_point}')

    def get_waypoints_path(self, waypoints: list):
        if len(waypoints) == 1:
            return waypoints
        path = []
        for i in range(len(waypoints) - 1):
            path += self.get_path(waypoints[i], waypoints[i + 1])
        return path
