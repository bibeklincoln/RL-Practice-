from gym.envs.registration import register

register(
    id="TicTacToeEnv-v0",
    entry_point="gym_tictac.envs:TicTacToeEnv",
)
