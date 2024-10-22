import pygame
import random
import time

pygame.init()

height = 700
width = 900

screen = pygame.display.set_mode((width,height))
pygame.display.set_caption('Catch The Grades!!')

black = (0,0,0)
white = (255,255,255)

clock = pygame.time.Clock()

bunny_width = 60 #box around the bunny so detect collsion later on

bunny = pygame.image.load('bunny.png')
F = pygame.image.load('F.png')
A = pygame.image.load('A+.png')



prefSize = (60, 60)
bunny = pygame.transform.scale(bunny, prefSize)
F = pygame.transform.scale(F, prefSize)
A = pygame.transform.scale(A, prefSize)  
    

def crash():
    time.sleep(5)
    pygame.quit()
    


def you(x,y):
    screen.blit(bunny, (x,y))

def F_grade(F_posX,F_posY,F_wid,F_height):
    screen.blit(F , (F_posX,F_posY))

def A_grade(A_posX,A_posY,A_wid,A_height):
    screen.blit(A , (A_posX,A_posY))

    
def run_game():
    x =  (width * 0.45)
    y = (height * .87)
    x_change = 0
    x_speed = 3

    F_startposX = random.randrange(0, width)
    F_startposY = -500
    F_speed = 5
    Fheight = 60
    Fwidth = 60
    

    A_startposX = random.randrange(0, width)
    A_startposY = -500
    A_speed = 5
    Aheight = 60
    Awidth = 60
    

    fail = False

    while not fail:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            #https://www.pygame.org/docs/ref/key.html (kedown stuff here)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_a:
                    x_change = -5
                elif event.key == pygame.K_d:
                    x_change = 5
            if event.type == pygame.KEYUP:
                if event.key == pygame.K_a or event.key == pygame.K_d:
                    x_change = 0
           
        x += x_change
               
        screen.fill(white)

        F_grade(F_startposX, F_startposY,Fwidth,Fheight)
        F_startposY += F_speed
        
        A_grade(A_startposX, A_startposY,Awidth,Aheight)
        A_startposY += F_speed

        you(x,y)
        
        if x > width - bunny_width or x < 0:
            crash()

        
        if A_startposY > height:
            A_startposX = random.randrange(0, width)
            A_startposY = 0 - Fheight
            

        if y < (F_startposY + bunny_width):
            if ((x < (F_startposX + bunny_width and x > F_startposX))
          or ((x + bunny_width) > F_startposX
           and (x + bunny_width) < (F_startposX + bunny_width))):
                crash()
            
        pygame.display.update()
        clock.tick(60)

run_game()
pygame.quit()
quit()
