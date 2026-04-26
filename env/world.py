import numpy as np

from env.constants import (
    CIVILIAN,
    DEFAULT_AGENT_VISION_RANGE,
    DEFAULT_FIRE_SPREAD_PROBABILITY,
    EMPTY,
    FIRE,
    UNKNOWN,
)


class World:
    def __init__(self, grid_size=10, fire_spread_probability=DEFAULT_FIRE_SPREAD_PROBABILITY):
        self.grid_size = grid_size
        self.fire_spread_probability = fire_spread_probability
        self.grid = np.zeros((grid_size, grid_size), dtype=np.int32)
        self.memory_grid = np.full((grid_size, grid_size), UNKNOWN, dtype=np.int32)

        self.civilians_to_save = 2
        self.civilians_saved = 0

        # Static Placements (Hazards and Civilians)
        self.grid[2][3] = FIRE
        self.grid[7][8] = FIRE
        self.grid[5][5] = CIVILIAN
        self.grid[1][8] = CIVILIAN

    def spread_fire(self, np_random):
        """Cellular automata logic for fire spreading."""
        new_fires = []
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                if self.grid[row][col] == FIRE:
                    neighbors = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    for dx, dy in neighbors:
                        nx, ny = row + dx, col + dy
                        if 0 <= nx < self.grid_size and 0 <= ny < self.grid_size:
                            if self.grid[nx][ny] == EMPTY:
                                # Use Gym's RNG for reproducible fire spread
                                if np_random.random() < self.fire_spread_probability:
                                    new_fires.append((nx, ny))
        for x, y in new_fires:
            self.grid[x][y] = FIRE

    def update_memory(self, explorers_positions, vision_range=DEFAULT_AGENT_VISION_RANGE):
        """Updates the shared global memory map based on agent vision."""
        for ex, ey in explorers_positions:
            for dx in range(-vision_range, vision_range + 1):
                for dy in range(-vision_range, vision_range + 1):
                    world_x, world_y = ex + dx, ey + dy
                    if 0 <= world_x < self.grid_size and 0 <= world_y < self.grid_size:
                        self.memory_grid[world_x][world_y] = self.grid[world_x][world_y]
