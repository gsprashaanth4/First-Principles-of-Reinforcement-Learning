import pygame
import numpy as np

class Visuals:
    def __init__(self, map, map_dim):
        pygame.init()
        self.map = map
        self.map_dim = map_dim

        self.width = map_dim
        self.height = map_dim
        self.scale = 100

        self.display_window = pygame.display.set_mode((self.width * self.scale, self.height * self.scale))
        self.virtual_canvas = pygame.Surface((self.width, self.height))

        self.arrow_list = []
        self.virtual_canvas.fill((30, 30, 30))

    def update(self, state, step):
        x = state[0]
        y = state[1]
        self.virtual_canvas.set_at((y, x), (40, 50, 170+step*2))

    def display(self, q_func):
        for event in pygame.event.get(pygame.QUIT):
            pygame.quit()
            raise SystemExit

        scaled_surface = pygame.transform.scale(self.virtual_canvas, self.display_window.get_size())
        self.display_window.blit(scaled_surface, (0, 0))
                    
        pygame.display.flip()

    def reset(self):
        self.virtual_canvas.fill((30, 30, 30))
        for x in range(self.map_dim):
            for y in range(self.map_dim):
                if self.map[x][y] == 1:
                    self.virtual_canvas.set_at((y, x), (160, 0, 0))
                if self.map[x][y] == 2:
                    self.virtual_canvas.set_at((y, x), (160, 160, 0))

    def wait_until_closed(self):
        waiting = True
    
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    waiting = False
    
            pygame.time.wait(10)
    
        pygame.quit()