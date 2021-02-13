# Objects in the game - Pipe, bord, ground
# create classes to represent the objects

import pygame
import neat
import time
import os
import random
pygame.font.init()

WIDTH = 550
HEIGHT = 900

STAT_FONT = pygame.font.SysFont("comicsans", 50)

# load the sequential bird images and scale them to twice its original size
BIRD_IMGS = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird1.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird2.png"))),
             pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bird3.png")))]

PIPE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "pipe.png")))
BASE_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "base.png")))
BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))

class Bird:
    IMGS = BIRD_IMGS
    MAX_ROT = 25            # how much the bird is going to tilt while going up and down
    ROT_VEL = 20            # how much the bird is rotating at each frame or every time we move the bird
    ANIMATION_TIME = 5      # how long we show each bird animaton, how fast bird will flap its wings

    def __init__(self, x, y):       # x, y => starting position for the bird
        self.x = x
        self.y = y
        self.tilt = 0               # how much the image is actually tilted, initally => bird will be flat
        self.tick_count = 0         # used to figure out the physics of bird
        self.vel = 0                # initial velocity = 0
        self.height = self.y
        self.img_count = 0          # which image of bird we are currently showing, so that we can animate and keep track
        self.img = self.IMGS[0]

    # Bird will flap up and jump upwards
    def jump(self):
        self.vel = -10.5            # upward => negative velocity, downward => positive, left => neg, right=>pos
        self.tick_count = 0         # keep track of when we last jumped.
        self.height = self.y        # where the bird originally jumped from, where it originally started moving from

    # move our bird at every single frame
    # calculate how much the bird object needs to move
    def move(self):
        self.tick_count+=1          # a frame went by => how many seconds we are moving for

        # displacement - how many pixels we are moving up or down the frame when changing y position of bird
        d = self.vel*self.tick_count + 1.5*(self.tick_count)**2
        # when tick count = 1 => d = -10.5*1 + 1.5*1 = -9 => move 9 pixels upwards

        # if moving down equal / more than 16 pixels => set d to 16 pixels do that we dont accelerate anymore
        if d>=16:
            d = 16

        if d<0:
            d-=2

        # Change the current y position so that we move slowly up/slowly down
        self.y = self.y+d

        if d<0 or self.y < self.height+50:
            # every time we jump => we keep track of where we jumped from
            # if bird position is above the previous jump position => still moving upwards, dont fall down yet
            # as soon as we get below the previous jump position => start tilting bird downwards

            # if moving upwards => dont want to tilt completely up and just tilt slightly
            if self.tilt < self.MAX_ROT:
                self.tilt = self.MAX_ROT    # immediately set the rotation of bird to 25 degrees
        else:
            # when we are moving down => want to tilt all the way to 90 degrees
            if self.tilt > -90:
                self.tilt -= self.ROT_VEL

    # for animation and drawing
    def draw(self, win):
        self.img_count += 1

        # check what image to show based on current image count
        # show flapping up and flapping down of the bird

        # if image count < 5 => show the first image
        if self.img_count <= self.ANIMATION_TIME:
            self.img = self.IMGS[0]

        # if image count < 10 => show the second image
        elif self.img_count <= self.ANIMATION_TIME*2:
            self.img = self.IMGS[1]

        # if image count < 15 => show the third image
        elif self.img_count <= self.ANIMATION_TIME*3:
            self.img = self.IMGS[2]
        elif self.img_count <= self.ANIMATION_TIME*4:
            self.img = self.IMGS[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMGS[0]
            self.img_count = 0

        # when bird tilted at 90 degrees going downwards => dont want flapping
        if self.tilt <= -80:
            self.img = self.IMGS[1]                 # image where wings are level and not flapping
            self.img_count = self.ANIMATION_TIME*2

        # Rotate image around the centre in pygame
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rect = rotated_image.get_rect(center = self.img.get_rect(topleft=(self.x, self.y)).center)
        win.blit(rotated_image, new_rect.topleft)

    # Check collision for objects
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

# Bird doesnt move but all the objects on the screen moves
class Pipe:
    GAP = 200       # space between pipe
    VEL = 5         # how fast the pipe in moving backward towards the bird

    def __init__(self, x):
        self.x = x
        self.height = 0
        self.gap = 100

        # where the top and bottom ends of the pipe is
        self.top = 0
        self.bottom = 0

        # TOP pipe => image of pipe which appears on the top => flip the image of bottom pipe
        # bottom pipe => image of pipe which appears on the bottom
        self.PIPE_TOP = pygame.transform.flip(PIPE_IMG, False, True)
        self.PIPE_BOTTOM = PIPE_IMG

        self.passed = False  # check if the bird has already passed the pipe

        # define where the top of the pipe is, where the bottom of pipe is and the gap
        self.set_height()

    def set_height(self):
        self.height = random.randrange(50, 450)             # random number where top of the pipe should be
        self.top = self.height - self.PIPE_TOP.get_height()
        self.bottom = self.height + self.GAP
        # figure out top left position of the image of pipe
        # for bottom pipe => the top left corner is exactly the 'y' of the pipe location

    # move the pipe x position based on the velocity in each frame
    # move the pipe to the left a little bit
    def move(self):
        self.x -= self.VEL

    # draw both the TOP pipe and the BOTTOM pipe
    def draw(self, win):
        win.blit(self.PIPE_TOP, (self.x, self.top))
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    # Easy way of collision
    # draw boxes around the object
    # check if boxes collide with each other => not accurate

    # Pixel-perfect collision using Mask-Collisions
    # Draw masks for pixel-perfect collision
    # Mask => 2D array/list of where all the pixels are located inside a box
    #      => 2D array => rows = number of pixels going down, cols = number of pixels going across
    # for both the objects colliding check in each list if there are any pixels which are in same area

    def collide(self, bird):
        bird_mask = bird.get_mask()                         # Get the mask

        # Create a mask for top pipe and bottom pipe
        top_mask = pygame.mask.from_surface(self.PIPE_TOP)
        bottom_mask = pygame.mask.from_surface(self.PIPE_BOTTOM)

        # offset => how far away are the masks(two top left corners) from each other
        # offset from bird to top mask
        # cant have decimal numbers => round off
        top_offset = (self.x - bird.x, self.top - round(bird.y))

        # offset from bird to bottom mask
        bottom_offset = (self.x - bird.x, self.bottom - round(bird.y))

        # finding point of collision to check if the masks collide
        # Point of collision between the bird mask and the bottom pipe using bottom_offset
        # if no collision => function returns None
        b_point = bird_mask.overlap(bottom_mask, bottom_offset)

        # Point of collision between the bird mask and the top pipe using top_offset
        t_point = bird_mask.overlap(top_mask,top_offset)

        # If any of b_point or t_point is not None => collision
        if b_point or t_point:
            return True

        # if we are not coliding both b_point and t_point will be None
        return False

# Platform / Floor of the game => needs to be constantly moving
# Image => not infinitely long
# As soon as it ends => need to move it back to starting point so that it can move again
# circle of 2 images moving again and again
# start of second image = end of 1st image
# keep replacing 1st image and 2nd image again and again to create moving effect
class Base:
    VEL = 5                         # Same velocity as pipe for similar speeds
    WIDTH = BASE_IMG.get_width()    # width of the image
    IMG = BASE_IMG

    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH

    # call on every single frame
    def move(self):
        self.x1 -= self.VEL
        self.x2 -= self.VEL

        # if one of the images goes off the screen completely
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH

        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH

    # Draw the base
    def draw(self, win):
        win.blit(self.IMG, (self.x1, self.y))
        win.blit(self.IMG, (self.x2, self.y))

# Draw on the window => pipes, bird, base
def draw_window(win, bird, pipes, base, score):
    win.blit(BG_IMG, (0, 0))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)
    bird.draw(win)

    text = STAT_FONT.render("Score: "+str(score), 1, (255, 255, 255))
    win.blit(text, (WIDTH-10-text.get_width(), 10))
    pygame.display.update()

def main():
    bird = Bird(230, 350)
    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    score = 0
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
        # bird.move()

        # Move the base
        base.move()

        # Move the pipe
        # Need to continuously create pipes as the game progresses

        add_pipe = False
        rem = []        # Pipes to remove
        for pipe in pipes:
            if pipe.collide(bird):
                pass

            # if pipe is completely off the screen => remove the pipe
            if pipe.x+pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            # check if we have passed the pipe
            # as soon as the bird passses a pipe => need to generate a new pipe for it to go through
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

            pipe.move()

        # once we pass a pipe => we got a point
        if add_pipe:
            score+=1
            pipes.append(Pipe(600))             # generate a new pipe and remove the previous pipe

        for r in rem:
            pipes.remove(r)

        # Bird hits the ground
        if bird.y+bird.img.get_height() >= 700:
            pass

        draw_window(win, bird, pipes, base, score)

    pygame.quit()
    quit()

main()
