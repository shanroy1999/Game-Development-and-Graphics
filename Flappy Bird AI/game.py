"""
Objects in the game - Pipe, bord, ground
create classes to represent the objects

NEAT algorithm => it is a genetic algorithm => NeuroEvolution of Augmenting Topologies
NEAT => evolves a neural network
Start => completely random without even knowing how the game works
after many generations of slow learning and slowly getting better
finds out patterns to figure out what to do to progress further
after few generations of slow learning => AI starts getting better exponentially until it cant be beaten

Input layer => inputs the information to the network
Input features for the input layer => Position of bird, position of top pipe, position of bottom pipe
Output layer => tell AI what to do => use 1 output neuron => tell whether to jump or not jump
Each input neuron => connected to output neuron through connections => each connection has its own weight
weight => number which represent how strong/weak connection is
tweak the weight to make AI better / worse.
bias => help adjust the network
weighted sum = sum(weight*input) + bias
apply tanh activation function to weighted sum => f(weighted_sum) => tanh(weighted_sum) => between -1 and 1
if o/p value of neural network > 0.5 => jump otherwise do not jump

NEAT => inspired by concept of natural selection
natural selection => process of generations getting better and better and better until they become best

flappy bird => no idea of what correct weights and biases should be => cant figure out without tons of tests
Start by creating a random population of birds => each population consist of 100 birds
neural network => control the bird in each population with random weights and random biases
Test all the neural networks on the game and see how well they do by evaluating fitness
fitness => depends on what game we play
Flappy bird => how well a bird did depends upon how far the bird is able to reach
fitness score = every frame the bird moves forward without dying / losing => get another point
end of simulation => all of the birds died => see which one performed the best in set of population
take those set of birds => breed them / mutate them to create a new population of 100 new birds
get rid of all birds that performed poorly
at last => we have the offsprings from the birds of last generation
offsprings => will perform better than last generation as they are generated from best of last generation

NEAT => changes the values of the connections / weights and randomly add other nodes and remove and add connections
Try to find the topology / architecture for the neural network and find the best one which works
NEAT => favour smaller/simpler architectures rather than too complex ones
inputs to neural network => a) position of bird(y position), b) distance between bird and next TOP_PIPE
                            c) distance between bird and next BOTTOM_PIPE

output of neural network => a) jump, b) not jump
activation function => let NEAT pick activation function for any hidden layers, use Tanh activation func
                    => Tanh => squishes values between -1 and 1
if >0.5 => jump else not jump
population size => arbitrary value in general => here, take 100 birds
[gen 0 => 100 birds] -> [gen 1 => 100 birds(mutated from the best of prev gen birds)] -> ....

fitness function => most important part in NEAT => how birds are going to get better
                 => evaluate how good the birds are
whichever bird goes furthest in the game => the best => fitness function will depend on distance travelled
max generation => max generations we have to run for => if we do not get the best one till it, try all over
               => 30 -> if we pass 30 generations and we still dont have perfect bird => try again all over

configuration file => very important while using neat module
                   => text file that sets up all the variables and parameters for NEAT algo to run

parameters :

[NEAT] section
1) fitness_criterion => min / max / mean => what function we use to determine the best birds
                     => max => take the birds that are best and breed those, remove those with lowest fitness
2) fitness_threshold => what fitness level do we want to get before termination of program
                     => 100 => if fitness score = 100 => no need to run the program further, no need to do anymore generations after
3) population_size => 100 => numer of individuals in each generation
4) reset_on_extinction => evaluates to True, when all species simultaneously become extinct, a new random population created
                       => if False => completeExtinctionException thrown

NEAT => all population members(i.e. birds) => called 'Genomes'
     => Genomes -> have nodes(ip node and op node) and genes(connections of nodes)

[DefaultGenome] Section
1) activation_default => default activation function assigned to new nodes
                      => tanh
2) activation_mutate_rate => change activation function randomly
                     => probability that mutation will generate new random activation function
                     => 0.0
3) activation_options => list of the activation functions that may be used by nodes
                      => by default -> sigmoid

node-bias option:
1) bias_max_value -> for the bias that we have what is the max value we can pick => 30.0
2) bias_min_value -> for the bias that we have what is the max value we can pick => -30.0

conn_add_prob , conn_delete_prob => how likely we are to add/remove connections
enabled_default => we can have connections that are sometimes enabled or not enabled
                => True -> connections are active
feed_forward => True -> we are using feed-forward network
initial_connection => full -> we are using fully connected neural network

node_add_prob, node_delete_prob => how likely we are to add/remove nodes

num_hidden, num_outputs, num_inputs => most important => setting default amount of input / hidden / output

Until the neural network reaches the fitness threshold => the game will continue generating
"""

import pygame
import random
import os
import time
import neat
import visualize
import pickle
pygame.font.init()  # init font

WIN_WIDTH = 600
WIN_HEIGHT = 800
FLOOR = 730
STAT_FONT = pygame.font.SysFont("comicsans", 50)
END_FONT = pygame.font.SysFont("comicsans", 70)
DRAW_LINES = True

WIN = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
pygame.display.set_caption("Flappy Bird")

# load the sequential bird images and scale them to twice its original size
bird_images = [pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","bird" + str(x) + ".png")))
               for x in range(1,4)]

pipe_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","pipe.png")).convert_alpha())
bg_img = pygame.transform.scale(pygame.image.load(os.path.join("imgs","bg.png")).convert_alpha(), (600, 900))
base_img = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs","base.png")).convert_alpha())

gen = 0

# Bird class representing the flappy bird
class Bird:
    IMGS = bird_images
    MAX_ROTATION = 25       # how much the bird is going to tilt while going up and down
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
        self.vel = -10.5            # upward => neg velocity, downward => pos, left => neg, right => pos
        self.tick_count = 0         # keep track of when we last jumped.
        self.height = self.y        # where the bird originally jumped from, where it originally started moving from

    # move our bird at every single frame
    # calculate how much the bird object needs to move
    def move(self):
        self.tick_count += 1        # a frame went by => how many seconds we are moving for

        # displacement - how many pixels we are moving up or down the frame when changing y position of bird
        # when tick count = 1 => d = -10.5*1 + 1.5*1 = -9 => move 9 pixels upwards
        displacement = self.vel*(self.tick_count) + 0.5*(3)*(self.tick_count)**2

        # if moving down equal / more than 16 pixels => set d to 16 pixels do that we dont accelerate anymore
        if displacement >= 16:
            displacement = (displacement/abs(displacement)) * 16

        if displacement < 0:
            displacement -= 2

        # Change the current y position so that we move slowly up/slowly down
        self.y = self.y + displacement

        if displacement < 0 or self.y < self.height + 50:
            # every time we jump => we keep track of where we jumped from
            # if bird position is above the previous jump position => still moving upwards, dont fall down yet
            # as soon as we get below the previous jump position => start tilting bird downwards

            # if moving upwards => dont want to tilt completely up and just tilt slightly
            if self.tilt < self.MAX_ROTATION:
                self.tilt = self.MAX_ROTATION  # immediately set the rotation of bird to 25 degrees

        else:  # tilt down
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
        blitRotateCenter(win, self.img, (self.x, self.y), self.tilt)

    # Check collision for objects
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

# In flappy bird, the bird doesn't move but the other objects move
# Move Pipe backwards or towards the bird to make it look like moving
class Pipe():
    GAP = 200           # Space between the pipe
    VEL = 10             # Velocity of the pipe

    # height of the pipes and where they show up on screen => completely random
    def __init__(self, x):
        self.x = x
        self.height = 0

        # where the top and bottom ends of the pipe is
        self.top = 0
        self.bottom = 0

        # TOP pipe => image of pipe which appears on the top => flip the image of bottom pipe
        # bottom pipe => image of pipe which appears on the bottom
        self.PIPE_TOP = pygame.transform.flip(pipe_img, False, True)
        self.PIPE_BOTTOM = pipe_img

        # check if the bird has already been passed by the pipe
        self.passed = False

        # define where the top of the pipe is, where the bottom of pipe is and the gap
        self.set_height()

    def set_height(self):

        # give random number where top of the pipe should be
        self.height = random.randrange(50, 450)

        # top left position of the TOP pipe image
        self.top = self.height - self.PIPE_TOP.get_height()

        # for bottom pipe => the top left corner is exactly the 'y' of the pipe location
        self.bottom = self.height + self.GAP

    # move the pipe x position based on the velocity in each frame
    # move the pipe to the left a little bit
    def move(self):
        self.x -= self.VEL

    # Draw both TOP Pipe and BOTTOM Pipe
    def draw(self, win):
        # draw TOP pipe
        win.blit(self.PIPE_TOP, (self.x, self.top))
        # draw BOTTOM pipe
        win.blit(self.PIPE_BOTTOM, (self.x, self.bottom))

    # Easy way of collision
    # draw boxes around the object
    # check if boxes collide with each other => not accurate

    # Pixel-perfect collision using Mask-Collisions
    # Draw masks for pixel-perfect collision
    # Mask => 2D array/list of where all the pixels are located inside a box
    #      => 2D array => rows = number of pixels going down, cols = number of pixels going across
    # for both the objects colliding check in each list if there are any pixels which are in same area

    def collide(self, bird, win):
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
    VEL = 10                         # Same velocity as pipe for similar speeds
    WIDTH = base_img.get_width()    # width of the image
    IMG = base_img

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

# Rotate image around the centre in pygame
def blitRotateCenter(surf, image, topleft, angle):
    rotated_image = pygame.transform.rotate(image, angle)
    new_rect = rotated_image.get_rect(center = image.get_rect(topleft = topleft).center)
    surf.blit(rotated_image, new_rect.topleft)

# Draw on the window => pipes, bird, base
def draw_window(win, birds, pipes, base, score, gen, pipe_ind):
    if gen == 0:
        gen = 1
    win.blit(bg_img, (0,0))

    # Draw all the pipes
    # pipes => list
    # cannot have more than 1 pipe at once
    for pipe in pipes:
        pipe.draw(win)

    base.draw(win)

    # draw all the birds
    for bird in birds:
        # draw lines from bird to pipe
        if DRAW_LINES:
            try:
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_TOP.get_width()/2, pipes[pipe_ind].height), 5)
                pygame.draw.line(win, (255,0,0), (bird.x+bird.img.get_width()/2, bird.y + bird.img.get_height()/2), (pipes[pipe_ind].x + pipes[pipe_ind].PIPE_BOTTOM.get_width()/2, pipes[pipe_ind].bottom), 5)
            except:
                pass
        # draw bird
        bird.draw(win)

    # Show Score on the window
    score_label = STAT_FONT.render("Score: " + str(score),1,(255,255,255))
    win.blit(score_label, (WIN_WIDTH - score_label.get_width() - 15, 10))

    # generations
    score_label = STAT_FONT.render("Gens: " + str(gen-1),1,(255,255,255))
    win.blit(score_label, (10, 10))

    # alive
    score_label = STAT_FONT.render("Alive: " + str(len(birds)),1,(255,255,255))
    win.blit(score_label, (10, 50))

    pygame.display.update()

# fitness function for the NEAT algorithm => eval_genomes
# fitness function => take all the genome and evaluates all of them
# run all the birds playing at the same time
def eval_genomes(genomes, config):
    global WIN, gen
    win = WIN
    gen += 1

    # Each position in the 3 lists correspond to the same bird
    # keep track of neural network that controls each bird
    nets = []

    # keep track of bird that the neural network is controlling
    birds = []

    # keep track of genome to change fitness based on how far they move
    ge = []

    # genome => tuple which has genome id and genome object => (id, object)

    for genome_id, genome in genomes:
        # initial fitness of bird = 0
        genome.fitness = 0

        # create the feed-forward netowork
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230,350))
        ge.append(genome)

    base = Base(FLOOR)
    pipes = [Pipe(700)]
    score = 0

    clock = pygame.time.Clock()

    run = True
    while run and len(birds) > 0:
        clock.tick(60)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
                break

        # pipe index = 0 => pipe we will look at for the input to the neural network
        pipe_ind = 0

        # if we have birds left
        if len(birds) > 0:

            # if we pass the pipes then change the pipe index to second pipe
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1

        # if no birds left => break

        # pass some values associated with the bird into the neural network and check if output value>0.5
        # if output value > 0.5 => make the bird jump
        for x, bird in enumerate(birds):
            # add fitness to bird for surviving this long for it to keep moving forward
            # for loop => run 30 times a second => every second bird is alive => gains 1 fitness point
            # => encourage bird to stay alive
            ge[x].fitness += 0.1
            bird.move()

            # activate neural network with inputs
            # input 1 => y coord of the bird
            # input 2 => distance between y coordinates of the top pipe and the bird
            # input 3 => distance between y coordinates of the bottom pipe and the bird
            output = nets[birds.index(bird)].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), abs(bird.y - pipes[pipe_ind].bottom)))

            # output => list of all the values of output neuron
            # In this case => we have only 1 output neuron => only 1 member in the list
            # check if output of neural network is greater than 0.5 => jump the bird
            if output[0] > 0.5:
                bird.jump()

        # Move the base
        base.move()

        # Need to continuously generate pipes as the game progresses
        rem = []            # Pipes to remove
        add_pipe = False
        for pipe in pipes:
            pipe.move()
            # check for collision
            for bird in birds:
                if pipe.collide(bird, win):
                    # if a bird hits a pipe and another bird is at same level and didnt hit the pipe
                    # we will favour the bird that didnt hit the pipe
                    # and decrease the fitness score of the bord that hit the pipe
                    ge[birds.index(bird)].fitness -= 1

                    # Remove the bird that hit the pipe, its neural network associated and the genome
                    nets.pop(birds.index(bird))
                    ge.pop(birds.index(bird))
                    birds.pop(birds.index(bird))

            # if pipe is completely off the screen => remove the pipe
            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            # check if we have passed the pipe
            # as soon as the bird passses a pipe => need to generate a new pipe for it to go through
            if not pipe.passed and pipe.x < bird.x:
                pipe.passed = True
                add_pipe = True

        # once we pass a pipe => we got a point
        if add_pipe:
            score += 1
            # can add this line to give more reward for passing through a pipe (not required)
            for genome in ge:
                genome.fitness += 5
            pipes.append(Pipe(WIN_WIDTH))       # generate a new pipe and remove the previous pipe

        for r in rem:
            pipes.remove(r)

        # If bird hit the ground or the top of the screen
        for bird in birds:
            if bird.y + bird.img.get_height() - 10 >= FLOOR or bird.y < -50:
                nets.pop(birds.index(bird))
                ge.pop(birds.index(bird))
                birds.pop(birds.index(bird))

        draw_window(WIN, birds, pipes, base, score, gen, pipe_ind)


def run(config_file):

    # define all the subheadings / properties used / set in the configuration file
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)       # set the path for configuration file to read

    # Create the population, which is the top-level object for a NEAT run.
    # Generate a population based on whatever we have in the configuration file
    p = neat.Population(config)

    # Stats reporters => give some output / progress / statistics about each generation / best fitness
    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.Checkpointer(5))

    # Set the fitness function that we are going to run for 50 generations
    # Run for up to 50 generations.
    # fitness function => set the fitness for our birds
    winner = p.run(eval_genomes, 50)

    # show final stats
    print('\nBest genome:\n{!s}'.format(winner))


if __name__ == '__main__':
    # Get path to the configuration file
    local_dir = os.path.dirname(__file__)       # Path to the directory we are currently in

    # Join local directory to name of configuration file
    config_path = os.path.join(local_dir, 'config-feedforward.txt')

    # load in the configuration file inside the run function
    run(config_path)
