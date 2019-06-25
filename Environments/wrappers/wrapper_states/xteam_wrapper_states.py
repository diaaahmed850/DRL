import os
import gym
from gym import spaces
from ple_xteam import PLE
import numpy as np

class PLEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True, ple_game=True, **kwargs):
        # set headless mode
        os.environ['SDL_VIDEODRIVER'] = 'dummy'
        # open up a game state to communicate with emulator
        import importlib
        if ple_game:
            game_module_name = ('ple_xteam.games.%s' % game_name).lower()
        else:
            game_module_name = game_name.lower()
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)(**kwargs)
        self.game_state = PLE(game, fps=30, display_screen=display_screen)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_height, self.screen_width = self.game_state.getScreenDims()
        #self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_width, self.screen_height, 3), dtype = np.uint8)
        low=np.full(self.game_state.getStateSize(),-1000)
        high=np.full(self.game_state.getStateSize(),1000)
        self.observation_space=spaces.Box(low=low, high=high, dtype=np.float32)
        self.viewer = None


    def _step(self, a):
        reward = self.game_state.act(self._action_set[a])
        #state = self._get_image()
        state=self.xteam_get_state()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}

    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(),3)) # Hack to fix the rotated image returned by ple
        return image_rotated
    def xteam_get_state(self):
        state_dict = self.game_state.getGameState()
        state = [state_dict[i] for i in state_dict]
        #return np.reshape(state, [1, len(state)])
        #print(np.array(state,dtype=np.float32).dtype)
        
        return np.array(state,dtype=np.float32)

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def _reset(self):
        low=np.full(self.game_state.getStateSize(),-1000)
        high=np.full(self.game_state.getStateSize(),1000)
        self.observation_space=spaces.Box(low=low, high=high, dtype=np.float32)
        self.game_state.reset_game()
        #state = self._get_image()
        state=self.xteam_get_state()

        return state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)


    def _seed(self, seed):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng

        self.game_state.init()