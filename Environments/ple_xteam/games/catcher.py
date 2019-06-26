import math
import sys
import random
import os 
import contextlib
with contextlib.redirect_stdout(None):
    import pygame
    from pygame.constants import K_a, K_d
    from .base.pygamewrapper import PyGameWrapper
    from .utils import percent_round_int



_dir_ = os.path.dirname(os.path.abspath(__file__))
_asset_dir = os.path.join(_dir_, "assets/")
class Paddle(pygame.sprite.Sprite):

    def __init__(self, speed, width, height, SCREEN_WIDTH, SCREEN_HEIGHT):
        self.speed = speed
        self.width = width

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.vel = 0.0

        pygame.sprite.Sprite.__init__(self)

        basket_path = os.path.join(_asset_dir, "basket.png")
        self.image = pygame.image.load(basket_path).convert_alpha()
        self.image = pygame.transform.scale(self.image,(int(SCREEN_WIDTH/6.85),int(SCREEN_HEIGHT/9.6)))
        
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect.center = (SCREEN_WIDTH / 2 - self.width / 2,SCREEN_HEIGHT - height)

    def update(self, dx, dt):
        self.vel += dx
        self.vel *= 0.8

        x, y = self.rect.center
        n_x = x + self.vel + 0.15

        if n_x <= 0:
            self.vel = 0.0
            n_x = 0

        if n_x >= self.SCREEN_WIDTH -  (self.width):
            self.vel = 0.0
            n_x = self.SCREEN_WIDTH - self.width


        self.rect.center = (n_x, y)

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)


class Fruit(pygame.sprite.Sprite):

    def __init__(self, speed, size, SCREEN_WIDTH, SCREEN_HEIGHT, rng):
        self.speed = speed
        self.size = size

        self.SCREEN_WIDTH = SCREEN_WIDTH
        self.SCREEN_HEIGHT = SCREEN_HEIGHT

        self.rng = rng

        pygame.sprite.Sprite.__init__(self)

        num = random.randrange(1,6,1)
        angle = random.randrange(0,180,30)

        fruit_path = os.path.join(_asset_dir, "fruit"+str(num)+".png")
        image = pygame.image.load(fruit_path).convert_alpha() 
        image = pygame.transform.scale(image,(int(self.SCREEN_HEIGHT/10),int(self.SCREEN_WIDTH/10)))
        image = pygame.transform.rotate(image,angle)

        self.image = image
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)
        self.rect.center = (-30, -30)

    def update(self, dt):
        x, y = self.rect.center
        n_y = y + self.speed * dt

        self.rect.center = (x, n_y)

    def reset(self,score):
        x = self.rng.choice(
            range(
                self.size *
                2,
                self.SCREEN_WIDTH -
                self.size *
                2,
                self.size))
        y = self.rng.choice(
            range(
                self.size,
                int(self.SCREEN_HEIGHT / 2),
                self.size))

        if(score % 10 == 0) and score > 6:
            self.speed += (self.SCREEN_HEIGHT*0.0001)
        if(self.speed >= 0.005*self.SCREEN_HEIGHT):
                self.speed = 0.005

        num = random.randrange(1,6,1)
        angle = random.randrange(0,180,30)
        fruit_path = os.path.join(_asset_dir, "fruit"+str(num)+".png")
        image = pygame.image.load(fruit_path).convert_alpha() 
        image = pygame.transform.scale(image,(int(self.SCREEN_HEIGHT/10),int(self.SCREEN_WIDTH/10)))
        image = pygame.transform.rotate(image,angle)
        self.image = image
        self.rect = self.image.get_rect()
        self.mask = pygame.mask.from_surface(self.image)

        self.rect.center = (x, -1 * y)

    def draw(self, screen):
        screen.blit(self.image, self.rect.center)


class Catcher(PyGameWrapper):
    """
    Based on `Eder Santana`_'s game idea.

    .. _`Eder Santana`: https://github.com/EderSantana

    Parameters
    ----------
    width : int
        Screen width.

    height : int
        Screen height, recommended to be same dimension as width.

    init_lives : int (default: 3)
        The number lives the agent has.

    """

    def __init__(self, width=480, height=480, init_lives=1):

        actions = {
            "left": K_a,
            "right": K_d
        }

        PyGameWrapper.__init__(self, width, height, actions=actions)

        self.fruit_size = percent_round_int(height, 0.06)
        self.fruit_fall_speed = 0.001*height#0.00095 * height
        self.state_size=4
        self.player_speed = 0.06 * width
        self.paddle_width = int(width/6.85)#percent_round_int(width, 0.2)
        self.paddle_height = int(height/9.6)#percent_round_int(height, 0.04)

        self.dx = 0.0
        self.init_lives = init_lives

    def _handle_player_events(self):
        self.dx = 0.0
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key

                if key == self.actions['left']:
                    self.dx -= self.player_speed

                if key == self.actions['right']:
                    self.dx += self.player_speed

    def init(self):
        self.score = 0
        self.lives = self.init_lives

        self.player = Paddle(self.player_speed, self.paddle_width,
                             self.paddle_height, self.width, self.height)

        self.fruit = Fruit(self.fruit_fall_speed, self.fruit_size,
                           self.width, self.height, self.rng)

        self.fruit.reset(0)

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
            "player_x": self.player.rect.center[0],
            "player_vel": self.player.vel,
            "fruit_x": self.fruit.rect.center[0],
            "fruit_y": self.fruit.rect.center[1]
        }

        return state

    def getScore(self):
        return self.score

    def game_over(self):
        return self.lives == 0

    def step(self, dt,width=480,height=480):
        
        self.screen.fill((0, 0, 0))
        self._handle_player_events()

        self.score += self.rewards["tick"]

        

        if self.fruit.rect.center[1] >= self.height:
            self.score += self.rewards["negative"]
            self.lives -= 1    
            self.fruit.reset(0.001*height)
        

        #hits = pygame.sprite.spritecollide(
            #self.player, self.terrain_group, False,pygame.sprite.collide_mask)

        if pygame.sprite.collide_rect(self.player, self.fruit):
            self.score += self.rewards["positive"]
            self.fruit.reset(self.getScore())

        self.player.update(self.dx, dt)
        self.fruit.update(dt)

        if self.lives == 0:
            self.score += self.rewards["loss"]
        background_path = os.path.join(_asset_dir, "background.jpg")
        background_image = pygame.image.load(background_path)#.convert_alpha()
        background_image = pygame.transform.scale(background_image,(width,height))
        self.screen.blit(background_image, [0, 0])
        self.player.draw(self.screen)
        self.fruit.draw(self.screen)

        score = int(self.getScore())
        if(score<0):
            score =0
        font = pygame.font.SysFont("Arial",int(width/15),True)
        font_surface = font.render(str(score),True,[0,0,0]) #"Score:"+
        self.screen.blit(font_surface , [int(width/2),int(height/12)])


if __name__ == "__main__":
    import numpy as np


    SCREEN_WIDTH = 480  
    SCREEN_HEIGHT = 480
    pygame.init()
    game = Catcher(SCREEN_WIDTH, SCREEN_HEIGHT)
    game.rng = np.random.RandomState(24)
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    logo_path = os.path.join(_asset_dir, "catcher.png")
    catcherImage = pygame.image.load(logo_path).convert_alpha()
    pygame.display.set_icon(catcherImage)
    pygame.display.set_caption("FruitCatcher")  
    game.clock = pygame.time.Clock()
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.reset()

        game.step(dt,SCREEN_WIDTH, SCREEN_HEIGHT)
        pygame.display.update()
