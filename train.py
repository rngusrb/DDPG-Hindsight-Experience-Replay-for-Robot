import numpy as np
import gymnasium as gym
import os, sys
from arguments import get_args
from mpi4py import MPI
from rl_modules.ddpg_agent import ddpg_agent
import random
import torch

"""
train the agent, the MPI part code is copy from openai baselines(https://github.com/openai/baselines/blob/master/baselines/her)

"""
def get_env_params(env):
    obs = env.reset()
    
    # close the environment
    params = {'obs': obs[0]['observation'].shape[0], #25
            'goal': obs[0]['desired_goal'].shape[0], #3
            'action': env.action_space.shape[0], #4
            'action_max': env.action_space.high[0], #1
            }
    
    params['max_timesteps'] = env._max_episode_steps # 60
    
    return params

def launch(args):
    
    # env = gym.make(args.env_name, render_mode = "human")
    
    
    # 환경을 만들고 ddpg agent를 생성한다.
    env = gym.make(args.env_name, max_episode_steps=60)
    
    # 각 mpi 프로세스의 rank를 이용해서 random seed를 설정한다.
    random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    np.random.seed(args.seed + MPI.COMM_WORLD.Get_rank())
    torch.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    if args.cuda:
        torch.cuda.manual_seed(args.seed + MPI.COMM_WORLD.Get_rank())
    #pick and place에 해당하는 observation, goal, action, action_max, max_timesteps를 가져와 해당되는 env_params를 가져온다.
    env_params = get_env_params(env)
    # 환경과 상호작용하기 위한 ddpg agent를 생성한다.
 #   ddpg_trainer = ddpg_agent(args, env, env_params, load_model_path="/home/daehun/hindsight-experience-replay/saved_models/FetchPickAndPlace1000" )
    ddpg_trainer = ddpg_agent(args, env, env_params )
    ddpg_trainer.learn()

if __name__ == '__main__':
    # ddpg+her를 수행하기 위해 병렬처리를 위한 스레드 설정.
    #OpenMP와 MKL에서 멀티스레딩을 사용하지 않도록 해서 각 라이브러리가 서로 간섭하지 않도록 함.
    
    os.environ['OMP_NUM_THREADS'] = '1' #Numpy와 같은 라이브러리에서 사용되는 Open MP의 스레드를 1로 제한.
    os.environ['MKL_NUM_THREADS'] = '1' #Intel의 Math Kernel Library의 스레드를 1로 제한.
    os.environ['IN_MPI'] = '1' #MPI환경에서 
    # get the params
    args = get_args()
    launch(args)
