#This cartpole example from https://www.codeproject.com/Articles/5271939/Cartpole-The-Hello-World-of-Reinforcement-Learning
#Need to install "pip3 install ray[rllib]==0.8.5" to work the code


import ray
from ray import tune
from ray.rllib.agents.dqn import DQNTrainer

ray.shutdown()
ray.init(
    include_webui=False,
    ignore_reinit_error=True,
    object_store_memory=8 * 1024 * 1024 * 1024  # 8GB limit … feel free to increase this if you can
)

ENV = 'CartPole-v0'
TARGET_REWARD = 195
TRAINER = DQNTrainer

tune.run(
     TRAINER,
     stop={"episode_reward_mean": TARGET_REWARD},  # stop as soon as we "solve" the environment
     config={
       "env": ENV,
       "num_workers": 0,  # run in a single process
       "num_gpus": 0,
       "monitor": True,  # store stats and videos periodically
       "evaluation_num_episodes": 25,  # every 25 episodes instead of the default 10
     }
)