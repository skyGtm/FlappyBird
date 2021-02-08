import pygame
import random
import os
import time
import neat
#Files Within Project
from base import Base
from bird import Bird
from pipe import Pipe


pygame.init()
pygame.font.init()

WIN_WIDTH = 500
WIN_HEIGHT = 800
GENERATION = 0

BG_IMG = pygame.transform.scale2x(pygame.image.load(os.path.join("imgs", "bg.png")))
STAT_FONT =  pygame.font.SysFont("comicsans", 50)


def draw_window(win, birds, pipes, base, score, GENERATION):
    win.blit(BG_IMG, (0,0))
    for pipe in pipes:
        pipe.draw(win)
    base.draw(win)

    for bird in birds:
        bird.draw(win)

    # Display Score
    text = STAT_FONT.render("Score: " + str(score), 1, (255, 255, 255))
    win.blit(text, (WIN_WIDTH - 10 - text.get_width(), 10))
    
    # Display Generation
    text = STAT_FONT.render("Generation: " + str(GENERATION), 1, (255, 255, 255))
    win.blit(text, (10, 10))
    
    pygame.display.update() 

    
def main(genomes, config):
    # bird = Bird(230, 350)
    global GENERATION
    nets = []
    ge = []
    birds = []
    GENERATION += 1


    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        birds.append(Bird(230, 350))
        genome.fitness = 0
        ge.append(genome)



    base = Base(730)
    pipes = [Pipe(600)]
    win = pygame.display.set_mode((WIN_WIDTH, WIN_HEIGHT))
    clock = pygame.time.Clock()
    score = 0
    
    run = True
    while run:
        clock.tick(30) # Increase to 60 if frame rate of game is low
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
        
        pipe_ind = 0
        if len(birds) > 0:
            if len(pipes) > 1 and birds[0].x > pipes[0].x + pipes[0].PIPE_TOP.get_width():
                pipe_ind = 1
        else:
            run = False
            break       
       
        for x, bird in enumerate(birds):
            bird.move()
            ge[x].fitness += 0.1

            output = nets[x].activate((bird.y, abs(bird.y - pipes[pipe_ind].height), \
                abs(bird.y - pipes[pipe_ind].bottom)))

            # print(output), output will be between -1 and 1 for given tanh activation function  
            if output[0] > 0.5:
                bird.jump()

        # bird.move()
        add_pipe = False
        rem = []
        for pipe in pipes:
            for x, bird in enumerate(birds):
                if pipe.collide(bird):
                    birds.pop(x)
                    nets.pop(x)
                    ge.pop(x)

                if not pipe.passed and (pipe.x + pipe.get_width()) < bird.x :
                    pipe.passed = True
                    add_pipe = True

            if pipe.x + pipe.PIPE_TOP.get_width() < 0:
                rem.append(pipe)

            pipe.move()

        # Score/ Add new pipe
        if add_pipe:
            score += 1
            for g in ge:
                g.fitness += 5
            pipes.append(Pipe(500))

        # Remove offscreen pipes
        for r in rem:
            pipes.remove(r)

        # Collision with ground at 730 or ceiling at 0
        for x, bird in enumerate(birds):
            if bird.y + bird.img.get_height() > 730 or bird.y < 0:
                birds.pop(x)
                nets.pop(x)
                ge.pop(x)
        
        base.move()
        draw_window(win, birds, pipes, base, score, GENERATION)
    


def run(config_file):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)
    p = neat.Population(config)

    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    # p.add_reporter(neat.CheckPointer(5))

    winner = p.run(main, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
        
