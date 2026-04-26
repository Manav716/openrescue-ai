from env.constants import (
    CIVILIAN,
    DEFAULT_DRONE_VISION_RANGE,
    DRONE_AGENT,
    EMPTY,
    FIRE,
    RESCUE_AGENT,
)


class BaseAgent:
    """Grid agent with bounded cardinal movement."""

    def __init__(self, start_pos, agent_id):
        self.pos = list(start_pos)
        self.agent_id = agent_id
        self.tile_under = EMPTY

    def get_new_position(self, action, grid_size):
        """Calculates the next position without actually moving yet."""
        x, y = self.pos
        if action == 0 and x > 0:
            x -= 1
        elif action == 1 and x < grid_size - 1:
            x += 1
        elif action == 2 and y > 0:
            y -= 1
        elif action == 3 and y < grid_size - 1:
            y += 1
        return [x, y]


class RescueAgent(BaseAgent):
    def __init__(self, start_pos):
        super().__init__(start_pos, agent_id=RESCUE_AGENT)


class DroneAgent(BaseAgent):
    def __init__(self, start_pos):
        super().__init__(start_pos, agent_id=DRONE_AGENT)

    def generate_messages(self, grid, grid_size, reported_locations, vision_range=DEFAULT_DRONE_VISION_RANGE):
        """Drone scans nearby tiles and generates communication alerts."""
        new_messages = []
        discovery_reward = 0
        drone_x, drone_y = self.pos

        for dx in range(-vision_range, vision_range + 1):
            for dy in range(-vision_range, vision_range + 1):
                world_x, world_y = drone_x + dx, drone_y + dy
                
                if 0 <= world_x < grid_size and 0 <= world_y < grid_size:
                    tile = grid[world_x][world_y]
                    location = (world_x, world_y)

                    if location in reported_locations:
                        continue

                    if tile == FIRE:
                        new_messages.append(f"Fire detected at ({world_x}, {world_y})")
                        reported_locations.add(location)
                    elif tile == CIVILIAN:
                        new_messages.append(f"Civilian detected at ({world_x}, {world_y})")
                        reported_locations.add(location)
                        discovery_reward += 20
                            
        return new_messages, discovery_reward
