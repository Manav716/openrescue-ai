import numpy as np

class World:
    def __init__(self, grid_size=10):
        self.grid_size = grid_size
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.memory_grid = np.full((grid_size, grid_size), -1, dtype=np.int32)

        self.civilians_to_save = 2
        self.civilians_saved = 0

        # Static Placements (Hazards and Civilians)
        self.grid[2][3] = 1
        self.grid[7][8] = 1
        self.grid[5][5] = 2
        self.grid[1][8] = 2

    def spread_fire(self, np_random):
        """Cellular automata logic for fire spreading."""
        new_fires = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row][col] == 1:
                    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    for dx, dy in neighbors:
                        nx, ny = row + dx, col + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if self.grid[nx][ny] == 0:
                                # Use Gym's RNG for reproducible fire spread
                                if np_random.random() < 0.10:
                                    new_fires.append((nx, ny))
        for x, y in new_fires:
            self.grid[x][y] = 1

    def update_memory(self, explorers_positions, vision_range=2):
        """Updates the shared global memory map based on agent vision."""
        for ex, ey in explorers_positions:
            for dx in range(-vision_range, vision_range + 1):
                for dy in range(-vision_range, vision_range + 1):
                    world_x, world_y = ex + dx, ey + dy
                    if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                        self.memory_grid[world_x][world_y] = self.grid[world_x][world_y]