from gym.envs.registration import registry, register, make, spec
from wrappers.wrapper_states.xteam_wrapper_states import PLEEnv

# Pygame
# ----------------------------------------
for game in ['Catcher','citycopter']:
    nondeterministic = False
    register(
        id='{}-v0'.format(game),
        entry_point='wrappers.wrapper_states.xteam_wrapper_states:PLEEnv',
        kwargs={'game_name': game, 'display_screen':True},
        tags={'wrapper_config.TimeLimit.max_episode_steps': 10000},
        nondeterministic=nondeterministic,
    )
