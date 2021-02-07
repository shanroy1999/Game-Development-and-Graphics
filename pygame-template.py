import pygame
import random

WIDTH = 360
HEIGHT = 480
FPS = 30

# define colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)

# initializing pygame and create window
pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Game Template")
clock = pygame.time.Clock()

# game loop
running = True
while(running):

    # keep loop running at right speed
    clock.tick(FPS)

    # Process Inputs(events)
    for event in pygame.event.get():

        # check for closing the window
        if(event.type==pygame.QUIT):
            running = False             # Game loop end

    # Update

    # Draw / render
    screen.fill(BLACK)

    # Double Buffering
    pygame.display.flip()       # Show the other side(flip) after drawing everything

pygame.quit()

# lag => update is so slow that the loop takes more than 1/30th of a second
