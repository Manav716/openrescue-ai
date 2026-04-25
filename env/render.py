import pygame

# Colors
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
YELLOW = (255, 255, 0)
WHITE = (255, 255, 255)

CELL_SIZE = 60
GRID_SIZE = 10
WINDOW_SIZE = CELL_SIZE * GRID_SIZE

class Renderer:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
        pygame.display.set_caption("OpenRescue AI Simulation")

    def draw(self, grid):
        # Keep Pygame responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.close()
                exit()

        self.screen.fill(BLACK)

        for row in range(GRID_SIZE):
            for col in range(GRID_SIZE):
                value = grid[row][col]
                color = BLACK

                if value == 1:
                    color = RED       # Fire
                elif value == 2:
                    color = GREEN     # Civilian
                elif value == 3:
                    color = BLUE      # Rescue Agent
                elif value == 4:
                    color = YELLOW    # Drone

                # Draw Cell
                pygame.draw.rect(
                    self.screen,
                    color,
                    (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                )

                # Draw Grid Lines
                pygame.draw.rect(
                    self.screen,
                    WHITE,
                    (col * CELL_SIZE, row * CELL_SIZE, CELL_SIZE, CELL_SIZE),
                    1
                )

        pygame.display.flip()

    def close(self):
        pygame.quit()