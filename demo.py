import torch
from rl_modules.models import actor
from arguments import get_args
import gymnasium as gym
import numpy as np

# process the inputs
def process_inputs(o, g, o_mean, o_std, g_mean, g_std, args):
    o_clip = np.clip(o, -args.clip_obs, args.clip_obs)
    g_clip = np.clip(g, -args.clip_obs, args.clip_obs)
    o_norm = np.clip((o_clip - o_mean) / (o_std), -args.clip_range, args.clip_range)
    g_norm = np.clip((g_clip - g_mean) / (g_std), -args.clip_range, args.clip_range)
    inputs = np.concatenate([o_norm, g_norm])
    inputs = torch.tensor(inputs, dtype=torch.float32)
    return inputs

if __name__ == '__main__':
    args = get_args()
    # 저장한 모델경로를 가져온다.
    # model_path = args.save_dir + args.env_name + '/model.pt'
    model_path = "/home/daehun/hindsight-experience-replay/saved_models/FetchPickAndPlace7000/model.pt"
    # 저장한 모델, 정규화 팩터 등을 가져온다.
    o_mean, o_std, g_mean, g_std, actor_model, critic_model = torch.load(model_path, map_location=lambda storage, loc: storage)
    # pick and place 환경을 생성한다. robot env에서 자동으로 terminated되는 것을 무시하기 위해
    env = gym.make(args.env_name, render_mode = "human", max_episode_steps=1000)
    # 환경을 초기화하여 observation을 획득후 actor를 초기화한다.
    observation = env.reset()
    # print("observation", observation)
    # get the environment params
    env_params = {'obs': observation[0]['observation'].shape[0], 
                  'goal': observation[0]['desired_goal'].shape[0], 
                  'action': env.action_space.shape[0], 
                  'action_max': env.action_space.high[0],
                  }
    #네트워크를 가져온다.
    actor_network = actor(env_params)
    actor_network.load_state_dict(actor_model)
    actor_network.eval()
    
    for i in range(args.demo_length):
        # 여러번의 데모마다 환경을 초기화한다.
        observation = env.reset()    
        # 처음 초기화되었을 때의 observation과 target인 goal g를 가져온다.
        obs = observation[0]['observation']
        g = observation[0]['desired_goal']
        terminated = False
        truncated = False
        #terminated나 truncated가 될 때까지 반복한다.
        while not terminated and not truncated:
        
            env.render()
            inputs = process_inputs(obs, g, o_mean, o_std, g_mean, g_std, args)
            with torch.no_grad():
                pi = actor_network(inputs)
            action = pi.detach().numpy().squeeze()
            # put actions into the environment
            observation_new, reward, terminated, truncated, info = env.step(action)
            # print("info", info)
            obs = observation_new['observation']
            g = env.getGoals()
        # print("truncated", truncated, "terminated", terminated)
        print('the episode is: {}, is success: {}'.format(i, info['is_success']))
