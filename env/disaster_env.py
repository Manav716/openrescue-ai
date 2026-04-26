import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.constants import (
    CIVILIAN,
    DEFAULT_FIRE_SPREAD_INTERVAL,
    DEFAULT_FIRE_SPREAD_PROBABILITY,
    DEFAULT_GRID_SIZE,
    DEFAULT_MAX_STEPS,
    DRONE_AGENT,
    EMPTY,
    FIRE,
    RESCUE_AGENT,
    UNKNOWN,
)
from env.world import World
from env.agents import RescueAgent, DroneAgent


class DisasterEnv(gym.Env):
    """OpenRescue AI Gymnasium environment.

    The learned policy controls the rescue unit. The drone follows an
    autonomous random policy and contributes natural-language observations.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 4}

    def __init__(
        self,
        grid_size=DEFAULT_GRID_SIZE,
        max_steps=DEFAULT_MAX_STEPS,
        fire_spread_probability=DEFAULT_FIRE_SPREAD_PROBABILITY,
        fire_spread_interval=DEFAULT_FIRE_SPREAD_INTERVAL,
        render_mode=None,
    ):
        super().__init__()
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.fire_spread_probability = fire_spread_probability
        self.fire_spread_interval = fire_spread_interval
        self.render_mode = render_mode
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: 10x10 Memory Map
        self.observation_space = spaces.Box(
            low=UNKNOWN, high=DRONE_AGENT, shape=(self.grid_size, self.grid_size), dtype=np.int32
        )
        self.world = None
        self.rescue_agent = None
        self.drone_agent = None
        self.current_step = 0
        self._messages = []
        self.reported_locations = set()
        self.explored_cells = set()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Initialize World and Agents
        self.world = World(self.grid_size, self.fire_spread_probability)
        self.rescue_agent = RescueAgent(start_pos=[0, 0])
        self.drone_agent = DroneAgent(start_pos=[self.grid_size - 1, self.grid_size - 1])
        
        self.current_step = 0
        
        # Place agents on the grid
        self._place_agent(self.rescue_agent)
        self._place_agent(self.drone_agent)
        
        # Data Tracking
        self._messages = []
        self.reported_locations = set()
        self.explored_cells = set()
        
        # Initial Memory Reveal
        self.world.update_memory([self.rescue_agent.pos, self.drone_agent.pos])
        
        return self.world.memory_grid.copy(), self._get_info()

    def _place_agent(self, agent):
        self.world.grid[agent.pos[0]][agent.pos[1]] = agent.agent_id

    def _move_agent(self, agent, action):
        """Helper to process grid movement for any agent."""
        # 1. Restore the tile the agent was standing on
        self.world.grid[agent.pos[0]][agent.pos[1]] = agent.tile_under
        
        # 2. Calculate and update to the new position
        agent.pos = agent.get_new_position(action, self.grid_size)
        
        # 3. Save the new tile it just stepped on
        agent.tile_under = self.world.grid[agent.pos[0]][agent.pos[1]]

    def _get_info(self):
        return {
            "step": self.current_step,
            "rescue_agent_position": tuple(self.rescue_agent.pos) if self.rescue_agent else None,
            "drone_agent_position": tuple(self.drone_agent.pos) if self.drone_agent else None,
            "civilians_saved": self.world.civilians_saved if self.world else 0,
            "civilians_to_save": self.world.civilians_to_save if self.world else 0,
            "messages": list(self._messages),
        }

    def step(self, action):
        self.current_step += 1
        reward = -1 
        done = False
        truncated = False
        
        # --- 1. RESCUE AGENT MOVEMENT ---
        self._move_agent(self.rescue_agent, action)
        
        # Exploration Reward
        current_cell = (self.rescue_agent.pos[0], self.rescue_agent.pos[1])
        if current_cell not in self.explored_cells:
            reward += 5
            self.explored_cells.add(current_cell)
        
        # Target Evaluation
        if self.rescue_agent.tile_under == CIVILIAN:
            reward += 100 
            self.world.civilians_saved += 1
            self.rescue_agent.tile_under = EMPTY
            
            if self.world.civilians_saved == self.world.civilians_to_save:
                reward += 200 
                done = True
        elif self.rescue_agent.tile_under == FIRE:
            reward -= 100 
            
        # Place agent back on grid for visualization
        self._place_agent(self.rescue_agent)
        
        # --- 2. DRONE LOGIC ---
        drone_action = self.np_random.integers(0, 4)
        self._move_agent(self.drone_agent, drone_action)
        self._place_agent(self.drone_agent)
        
        # Drone Communication
        new_msgs, disc_reward = self.drone_agent.generate_messages(
            self.world.grid, self.grid_size, self.reported_locations
        )
        self._messages.extend(new_msgs)
        reward += disc_reward
        
        # --- 3. ENVIRONMENT UPDATES ---
        if self.current_step % self.fire_spread_interval == 0:
            self.world.spread_fire(self.np_random)
            
        if self.current_step >= self.max_steps: 
            truncated = True
            
        self.world.update_memory([self.rescue_agent.pos, self.drone_agent.pos])
        
        return self.world.memory_grid.copy(), reward, done, truncated, self._get_info()

    # --- Properties to maintain compatibility with main.py and render.py ---
    @property
    def grid(self):
        return self.world.grid

    @property
    def messages(self):
        return self._messages

    def render(self):
        if self.render_mode == "ansi":
            return str(self.grid)
        print(self.grid)
