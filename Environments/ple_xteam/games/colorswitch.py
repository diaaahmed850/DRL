import math
import sys
import random
import os 
import contextlib
import numpy as np

with contextlib.redirect_stdout(None):
    import pygame
    import pygame.gfxdraw
    from pygame.constants import K_a, K_SPACE
    from .base.pygamewrapper import PyGameWrapper

pygame.init()
pygame.font.init()
SCREEN_WIDTH, SCREEN_HEIGHT = 500,700
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Color Switch")
clock = pygame.time.Clock()
obstacles = []#list()
stars     = []    #list()
pies      = []     #list()
score = 0 
temp_score = 0
font = pygame.font.Font(pygame.font.get_default_font(), 24)
menu_font = pygame.font.Font(pygame.font.get_default_font(), 60)

PURPLE = (140, 0, 255)    #GREYSCALE 	(70, 70, 70)
RED    = (255, 12, 150)    #GREYSCALE 	(100, 100, 100)
TEAL   = (45, 220, 240)   #GREYSCALE    (170, 170, 170)
YELLOW = (250, 225, 20)   #GREYSCALE 	(210, 210, 210)
BLACK  = (0, 0, 0)
WHITE  = (255, 255, 255)
#colors = [PURPLE,RED,TEAL,YELLOW]
colors = [TEAL,YELLOW,PURPLE,RED]
angles = "0-90 TEAL , 90-180 YELLOW, 180-270 PURBLE , 270-360 RED"

def get_colorIndex(angle):
    if angle <= 90:
        return 0
    if 180 > angle > 90:
        return 1
    if 270 > angle >= 180:
        return 2
    if 360 > angle >= 270:
        return 3

def random_color():
    rand = random.randint(0,3)
    return colors[rand]


class Position:
    x = 0
    y = 0
    def __init__(self):
        self.x = 0
        self.y = 0

cam = Position()

class Obstacle(pygame.sprite.Sprite):

    def __init__(self, surface, x=250, y=150, rad=220, angle = 0, vel = 1):
        self.x = x
        self.y = y
        self.rad = rad
        self.angle = angle
        self.surface = surface
        self.vel = vel
        self.thickness = 25
        self.y_pos = 0

    def update(self):
        x, self.y_pos = (self.x-float(self.rad/2)-cam.x, self.y-float(self.rad/2)-cam.y)
        if(self.y_pos >= SCREEN_HEIGHT):
            obstacles.remove(self)
            #print("obstacle was removed")
            return
        self.angle+=self.vel
        #print(self.angle)
        if(self.angle > 360):
            self.angle-=360
        elif(self.angle <= 0):
            self.angle+=360

    
    def draw(self):
        #print(cam.x,cam.y)
        x, self.y_pos = (self.x-float(self.rad/2)-cam.x, self.y-float(self.rad/2)-cam.y)
        x, self.y_pos = int(x), int(self.y_pos)
        thick = self.thickness
        pygame.draw.arc(self.surface, PURPLE , (x, self.y_pos, self.rad, self.rad), math.radians(0+self.angle) ,math.radians(90+self.angle), thick)
        pygame.draw.arc(self.surface, PURPLE , (x, self.y_pos+1, self.rad, self.rad), math.radians(0+self.angle) ,math.radians(90+self.angle), thick)
        pygame.draw.arc(self.surface, YELLOW , (x, self.y_pos, self.rad, self.rad), math.radians(90+self.angle) , math.radians(180+self.angle), thick)
        pygame.draw.arc(self.surface, YELLOW , (x, self.y_pos+1, self.rad, self.rad), math.radians(90+self.angle) ,math.radians(180+self.angle), thick)
        pygame.draw.arc(self.surface, TEAL , (x, self.y_pos, self.rad, self.rad), math.radians(180+self.angle) ,math.radians(270+self.angle), thick)
        pygame.draw.arc(self.surface, TEAL , (x, self.y_pos+1, self.rad, self.rad), math.radians(180+self.angle) ,math.radians(270+self.angle), thick)
        pygame.draw.arc(self.surface, RED , (x, self.y_pos, self.rad, self.rad), math.radians(270+self.angle) ,math.radians(360+self.angle), thick)
        pygame.draw.arc(self.surface, RED , (x, self.y_pos+1, self.rad, self.rad), math.radians(270+self.angle) ,math.radians(360+self.angle), thick)
        pygame.gfxdraw.aacircle(self.surface, int(self.x-cam.x), int(self.y-cam.y), int(self.rad/2)+1, (20,20,20))
        pygame.gfxdraw.aacircle(self.surface, int(self.x-cam.x), int(self.y-cam.y), int(self.rad/2), (20,20,20))
        pygame.gfxdraw.aacircle(self.surface, int(self.x-cam.x), int(self.y-cam.y), int(self.rad/2)-thick-1, (20,20,20))
        pygame.gfxdraw.aacircle(self.surface, int(self.x-cam.x), int(self.y-cam.y), int(self.rad/2)-thick, (20,20,20))


class Star(pygame.sprite.Sprite):
    def __init__(self, surface, x, y):
        self.x = x
        self.y = y
        self.w = 10
        self.h = 10
        self.surface = surface
        self.color = WHITE
        self.dead = False

    def update(self):
        if(self.dead):
            stars.remove(self)

    def draw(self):
        x,y = self.x-cam.x,self.y-cam.y
        if(not self.dead):
            points = ((x,y-16),(x-7,y-5), (x-20,y-3), (x-11,y+8), (x-13, y+21), (x, y+16), (x+13, y+21), (x+11, y+8), (x+20, y-3), (x+7,y-5))
            pygame.gfxdraw.aapolygon(self.surface, points, self.color)
            pygame.gfxdraw.filled_polygon(self.surface, points, self.color)
        
        """
        This is commented for screenshots training
        """
        #else:
            #self.surface.blit(font.render("+1", True, (255-self.dead_counter*5, 255-self.dead_counter*5, 255-self.dead_counter*5)), (x-10,y-self.dead_counter))


        # pygame.draw.polygon(screen, RED, points)

class Pie(pygame.sprite.Sprite):
    def __init__(self, surface, x, y ,color = random_color()):
        self.x = x
        self.y = y
        self.surface = surface
        self.rad = 22
        self.color = color

    def draw_pie(self,x, y, rad, s_angle, e_angle, color):
        points = [(x,y)]
        for n in range(s_angle, e_angle+1):
            tx = x + int(rad*math.cos(math.radians(n)))
            ty = y + int(rad*math.sin(math.radians(n)))
            points.append((tx, ty))
        points.append((x,y))
        if(len(points)>2):
            pygame.gfxdraw.aapolygon(screen, points, color)
            pygame.gfxdraw.filled_polygon(screen, points, color)

    def draw(self):
        x, y = int(self.x-cam.x), int(self.y-cam.y)
        self.draw_pie(x, y, self.rad, 0, 90, RED)
        self.draw_pie(x, y, self.rad, 90, 180, TEAL)
        self.draw_pie(x, y, self.rad, 180, 270, YELLOW)
        self.draw_pie(x, y, self.rad, 270, 360, PURPLE)
        pygame.gfxdraw.aacircle(self.surface, x, y,self.rad-1, (20,20,20))
        pygame.gfxdraw.aacircle(self.surface, x, y,self.rad, (20,20,20))


class Ball(PyGameWrapper):


    def __init__(self, surface, x=250, y=400):
        self.x = x
        self.y = y
        self.rad = 10
        self.surface = surface
        self.vel = 0
        self.color = random_color()
        self.dead = False
        self.x_pos = 0
        self.y_pos = 0

    def cam_score(self):
        return int(abs(cam.y)/50)


    def collision_detection(self):
        global score
        self.x_pos, self.y_pos = self.x-cam.x, self.y-cam.y
        for star in stars:
            if(star.y+16 >= self.y):
                star.color = BLACK
                if(not star.dead):
                    score+=10
                star.dead = True
            
        #print(y)
        if(self.y_pos>=SCREEN_HEIGHT):
            self.die()

        for obstacle in obstacles:
            if(obstacle.y+int(obstacle.rad/2) >= self.y and obstacle.y+int(obstacle.rad/2)-25 <= self.y):
                if(self.color != YELLOW and obstacle.angle > 90 and obstacle.angle <= 180):
                    self.die()
                elif(self.color != PURPLE and obstacle.angle > 180 and obstacle.angle <= 270):
                    self.die()
                elif(self.color != RED and obstacle.angle > 270 and obstacle.angle <= 360):
                    self.die()
                elif(self.color != TEAL and obstacle.angle <= 90):
                    self.die()

            elif(obstacle.y-(obstacle.rad/2)+25 >= self.y-self.rad and obstacle.y-(obstacle.rad/2) <= self.y):
                if(self.color != RED and obstacle.angle > 90 and obstacle.angle <= 180):
                    self.die()
                elif(self.color != TEAL and obstacle.angle > 180 and obstacle.angle <= 270):
                    self.die()
                elif(self.color != YELLOW and obstacle.angle > 270 and obstacle.angle <= 360):
                    self.die()
                elif(self.color != PURPLE and obstacle.angle <= 90):
                    self.die()

        for cs in pies:
            if(cs.y >= self.y-self.rad*2):
                self.color = cs.color
                pies.remove(cs)
                self.color = random_color()
                

    def die(self):
        self.dying_counter = 0
        self.dead = True


    def update(self):
        if(not self.dead):
            self.vel -= 0.5
            self.y -= self.vel
            if(cam.y >= self.y-SCREEN_HEIGHT/2):
                cam.y = self.y-SCREEN_HEIGHT/2
            self.collision_detection()
        
    def draw(self):
        x = int(self.x-cam.x)
        y = int(self.y-cam.y)

        if(not self.dead):
            pygame.gfxdraw.aacircle(self.surface, x, y, self.rad, self.color)
            pygame.gfxdraw.filled_circle(self.surface, x, y, self.rad, self.color)
        else:
            #restart()
            pass



def restart():
    global cam, obstacles, score, stars
    global ball
    cam = Position()
    ball = Ball(screen)
    del stars[:]
    del obstacles[:]
    del pies[:]
    for i in range(50):
        #o_type = random.randint(0,2)
        o_type = 0
        if(o_type == 0):
            temp = Obstacle(screen, SCREEN_WIDTH/2, -600*i)
            obstacles.append(temp)
            temp_pie = Pie(screen, SCREEN_WIDTH/2, -600*i+300)
        """
        elif(o_type == 1):
            print(o_type)
            temp = Obstacle(screen, SCREEN_WIDTH/2, -600*i, 300,45,1)
            temp2 = Obstacle(screen, SCREEN_WIDTH/2, -600*i, temp.rad-temp.thickness*2-5, 180+45, -1)
            obstacles.append(temp)
            obstacles.append(temp2)
            t = random.randint(0,1)
            if(t == 0):
                col = RED
            else:
                col = YELLOW
            temp_pie = Pie(screen, SCREEN_WIDTH/2, -600*i+300, col)
        elif(o_type == 2):
            temp = Obstacle(screen, SCREEN_WIDTH/2-100, -600*i, 200, 45, 1)
            temp2 = Obstacle(screen, SCREEN_WIDTH/2+100, -600*i, 200, 45, -1)
            obstacles.append(temp)
            obstacles.append(temp2)
            temp_pie = Pie(screen, SCREEN_WIDTH/2, -600*i+300, TEAL)
        """
        temp_star = Star(screen, SCREEN_WIDTH/2, -600*i)
        stars.append(temp_star)
        pies.append(temp_pie)

    score = -10




class colorswitch(PyGameWrapper):
   

    def __init__(self, width=500, height=700, init_lives=1):

        actions = {
<<<<<<< HEAD
            "UP": K_SPACE
=======
            "UP": K_SPACE,
            #"None": K_a
>>>>>>> d778cef2210363fd6ef04cd97aff727c66f1dbe0
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)

        self.state_size = 2
        self.init_lives = init_lives

        self.surface = screen
        self.x = 250
        self.y = 400
        self.rad = 220
        self.angle = 0
        self.vel = 1
        self.color = random_color() 

    def _handle_player_events(self):
        for e in pygame.event.get():
            if(e.type == pygame.QUIT):
                pygame.quit()
                sys.exit()
                #return False
            if(e.type == pygame.KEYDOWN):
                if(e.key == pygame.K_ESCAPE):
                    pygame.quit()
                    sys.exit()
                    #return False
                elif(e.key == pygame.K_SPACE):
                    ball.vel = 8

    def init(self):
        global score
        score = 0
        self.lives = self.init_lives

        self.Ball = Ball(self.surface,self.x,self.y)

        self.Obstacle = Obstacle(self.surface, self.x, self.y, self.rad, self.angle , self.vel)

        self.Pie = Pie(self.surface, self.x, self.y ,self.color)

        self.Star = Star(self.surface,self.x,self.y)

        #self.fruit.reset(0)

    def getGameState(self):
        """
        Gets a non-visual state representation of the game.

        Returns
        -------

        dict
            * player x position.
            * players velocity.
            * fruits x position.
            * fruits y position.

            See code for structure.

        """
        state = {
            "player_color":colors.index(ball.color),
            "obstacle_color":get_colorIndex(obstacles[0].angle)
        }

        return state


    def reset(self):
        restart()
        #self.init()
        
    def game_over(self):
        return ball.dead
    
    #temp_score = 0
    def getScore(self):
        global score,temp_score
        if ((ball.cam_score() - temp_score) != 0):
            score += 1
            temp_score = ball.cam_score()
        #print(ball.cam_score())
        return score 


    def step(self,width=500,height=700):
        global score
        
        #self.screen.fill((0, 0, 0))
        self._handle_player_events()

        score += self.rewards["tick"]
        clock.tick(80)
        screen.fill((20,20,20))
        #print(len(obstacles))
        for obstacle in obstacles:
            obstacle.update()
        ball.update()
        for star in stars:
            star.update()

        for obstacle in obstacles:
            if(obstacle.y+obstacle.rad/2-cam.y >= 0 and obstacle.y-obstacle.rad/2-cam.y <= SCREEN_HEIGHT):
                obstacle.draw()
        for star in stars:
            if(star.y+13-cam.y >= 0 and star.y-13-cam.y <= SCREEN_HEIGHT):
                star.draw()
        for cs in pies:
            if(cs.y+cs.rad-cam.y >= 0 and cs.y-cs.rad-cam.y <= SCREEN_HEIGHT):
                cs.draw()
        ball.draw()
        #screen.blit(font.render(str(score), True, WHITE), (10, 10))


if __name__ == "__main__":
    import numpy as np


    
    restart()
    SCREEN_WIDTH = 500  
    SCREEN_HEIGHT = 700
    pygame.init()
    game = colorswitch(SCREEN_WIDTH, SCREEN_HEIGHT)
    #game.rng = np.random.RandomState(24)
    #game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    #logo_path = os.path.join(_asset_dir, "catcher.png")
    logo_path = "color.jpg"
    #colorImage = pygame.image.load(logo_path).convert_alpha()
    #pygame.display.set_icon(colorImage)
    pygame.display.set_caption("colorswitch")  
    game.clock = pygame.time.Clock()
    game.init()

    while True:
        if game.game_over():
            game.reset()
            #score += -5

        game.step(SCREEN_WIDTH, SCREEN_HEIGHT)
        print(game.getScore())
        #pygame.display.flip()
        pygame.display.update()
