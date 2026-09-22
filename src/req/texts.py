import pygame

class TextBox:
    def __init__(self, x, y, width, height, text_color, screen):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.text_color = text_color
        self.screen = screen

    def display(self, message, text_size, box_color):
        self.rect = pygame.Rect(self.x, self.y, self.width, self.height)
        pygame.draw.rect(self.screen, box_color, self.rect)

        font = pygame.font.SysFont('Arial', text_size)
        text_surface = font.render(message, True, self.text_color)
        text_rect = text_surface.get_rect(center=self.rect.center)
        self.screen.blit(text_surface, text_rect)