import torch
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.replay_buffer import replay_buffer
from rl_modules.models import actor, critic
from mpi_utils.normalizer import normalizer
from her_modules.her import her_sampler

"""
ddpg with HER (MPI-version)

"""
class ddpg_agent:
    def __init__(self, args, env, env_params, load_model_path = False):
        self.args = args
        self.env = env
        self.env_params = env_params
        #ddpg에서 사용할 actor와 critic을 초기화.
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        #각 네트워크를 멀티프로세스 상에서 동기화
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        #ddpg의 구현을 위해 타겟 네트워크 초기화
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        # 타겟 네트워크를 초기화
        if load_model_path:
            # print(torch.load(load_model_path + '/model.pt'))
            o_norm_mean, o_norm_std, g_norm_mean, g_norm_std, actor_state_dict, critic_state_dict = torch.load(load_model_path + '/model.pt')
            
            self.actor_network.load_state_dict(actor_state_dict)
            self.actor_target_network.load_state_dict(actor_state_dict)
            self.critic_network.load_state_dict(critic_state_dict)
            self.critic_target_network.load_state_dict(critic_state_dict)
            
            
        else:
            self.actor_target_network.load_state_dict(self.actor_network.state_dict())
            self.critic_target_network.load_state_dict(self.critic_network.state_dict())
            
        # cuda
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
        # 각 네트워크 파라미터에 대해 옵티마이저 초기화
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        #her 모듈 초기화
        self.her_module = her_sampler(self.args.replay_strategy, self.args.replay_k, self.env.compute_reward)
        # replay buffer 초기화
        self.buffer = replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # normalizer 초기화
        self.o_norm = normalizer(size=env_params['obs'], default_clip_range=self.args.clip_range)
        self.g_norm = normalizer(size=env_params['goal'], default_clip_range=self.args.clip_range)
        if load_model_path:
            self.o_norm.mean = o_norm_mean
            self.o_norm.std = o_norm_std
            self.g_norm.mean = g_norm_mean
            self.g_norm.std = g_norm_std
        # 저장파일이 존재하지 않는다면 마스터 프로세스만 저장 디렉토리를 생성한다.
        if MPI.COMM_WORLD.Get_rank() == 0:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            self.model_path = os.path.join(self.args.save_dir, self.args.env_name)
            if not os.path.exists(self.model_path):
                os.mkdir(self.model_path)

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles): #에포크당 샘플을 모으기 위한 에피소드 :50
                mb_obs, mb_ag, mb_g, mb_actions = [], [], [], []#각 에피소드마다 버퍼에 mini-batch obs, achieved goal, goal, actions를 저장한다.
                for _ in range(self.args.num_rollouts_per_mpi):#에피소드당 샘플을 모으기 위한 mpi마다의 rollout : 2 즉 미니배치 사이즈.
                    #미니배치에 추가할 에피소드별 obs, achieved goal, goal, actions를 저장한다.
                    ep_obs, ep_ag, ep_g, ep_actions = [], [], [], []
                    #환경을 리셋한다.
                    observation = self.env.reset()
                    
                    
                    obs = observation[0]['observation']
                    ag = observation[0]['achieved_goal']
                    g = observation[0]['desired_goal']
                    # max_timesteps만큼 반복하면서 에피소드를 수행한다.
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            #현재 상태 obs와 goal g를 이용해 input tensor를 만든다. 즉, obs와 goals를 같이 입력으로 넣어준다. == HER
                            input_tensor = self._preproc_inputs(obs, g)
                            #actor 네트워크를 이용해 action을 선택한다.
                            pi = self.actor_network(input_tensor)
                            action = self._select_actions(pi)
                        
                        # print("self.env.step(action):", self.env.step(action))
                        #선택한 행동을 환경에 넣어 다음 상태를 얻는다.
                        observation_new, _, _, _, info = self.env.step(action)
                        #새로운 상태.
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # 각 mpi에서 수행한 에피소드별 obs, achieved goal, goal, actions를 저장한다.
                        ep_obs.append(obs.copy()) #현재 상태
                        ep_ag.append(ag.copy()) #achived goal
                        ep_g.append(g.copy()) #goal
                        ep_actions.append(action.copy()) #action
                        # re-assign the observation
                        obs = obs_new
                        ag = ag_new
                    ep_obs.append(obs.copy())
                    ep_ag.append(ag.copy())
                    #mpi의 현재 rollout에서 수행한 에피소드를 미니배치버퍼에 추가한다.
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                # 저장한 에피소드들을 numpy array로 변환한다.
                mb_obs = np.array(mb_obs)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                #numpy array를 replay buffer에 저장한다.
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions])
                #normalizer를 업데이트한다.
                self._update_normalizer([mb_obs, mb_ag, mb_g, mb_actions])
                for _ in range(self.args.n_batches):
                    # 배치사이즈만큼 네트워크를 업데이트한다.
                    self._update_network()
                # 안정적인 네트워크 업데이트를 위해 ddpg에서는 main network를 업데이트하다가 일정 주기마다 target network를 업데이트한다.
                #즉, 미니배치만큼 네트워크를 업데이트하고 그 이후 어느정도 policy와 advatage의 차이가 벌어지기 전 target network를 업데이트한다.
                self._soft_update_target_network(self.actor_target_network, self.actor_network)
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            #agent를 evaluation하여 success rate를 구한다.
            success_rate = self._eval_agent()
            if MPI.COMM_WORLD.Get_rank() == 0: #프로세스의 랭크가 0일때, 즉 마스터 프로세스일때
                print('[{}] epoch is: {}, eval success rate is: {:.3f}'.format(datetime.now(), epoch, success_rate))
                torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std, self.actor_network.state_dict(), self.critic_network.state_dict()], \
                            self.model_path + '/model.pt')

    # 입력 전처리. her구현을 위해 obs와 goal을 같이 넣어준다.
    def _preproc_inputs(self, obs, g):
        obs_norm = self.o_norm.normalize(obs) #ddpg에서는 obs를 normalize한다.
        g_norm = self.g_norm.normalize(g)#ddpg에서는 goal을 normalize한다.
        #정규화된 observation과 goal을 같이 넣어준다.
        inputs = np.concatenate([obs_norm, g_norm])
        inputs = torch.tensor(inputs, dtype=torch.float32).unsqueeze(0)
        if self.args.cuda:
            inputs = inputs.cuda()
        return inputs
    #네틍워크를 업데이트하기 액션을 선택한다. exploration을 위해 action에 noise를 추가한다.
    
    def _select_actions(self, pi):
        action = pi.cpu().numpy().squeeze()
        # 가우시안 노이즈를 더해주는데, action의 최대값에서 noise_eps만큼 노이즈를 추가한다.
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        # action을 clip한다.
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        #epsilon의 확률로 랜덤 액션을 고르게 하기 위해서 베르누이 분포를 통해 랜덤 액션을 선택한다.
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action) 
        return action

    # normalizer를 업데이트한다.
    def _update_normalizer(self, episode_batch):
        #저장한 에피소드 배치에서 각각의 미니배치 observation, achived goal, goal, actions를 가져온다.
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        
        
        #모든 에피소드에 대해서, 첫번쨰 time step을 제외해서 다음 time step으로 대응시킨다.
        mb_obs_next = mb_obs[:, 1:, :] 
        mb_ag_next = mb_ag[:, 1:, :]
        
        
        # 한 에피소드에서 일어난 timesteps의 수, 즉 한 에피소드에서의 상태전이(trainsition)의 수를 가져온다.
        """
        self.args.num_rollouts_per_mpi = 2: 각 MPI 프로세스가 수행하는 에피소드의 수 : 2
    •	self.env_params['max_timesteps'] = 50: 각 에피소드의 timestep 수 : 50
    •	self.env_params['action'] = 4: pick and place 환경의 action 차원 : 4
        """
        num_transitions = mb_actions.shape[1]
        
        
        
        #buffer에 저장하기 위한 모든 정보를 담는다.
        buffer_temp = {'obs': mb_obs, 
                       'ag': mb_ag,
                       'g': mb_g, 
                       'actions': mb_actions, 
                       'obs_next': mb_obs_next,
                       'ag_next': mb_ag_next,
                       }
        #her을 통해 미래의 achived_goal을 현재 goal로 update하고, transition 대체, reward 재계산
        transitions = self.her_module.sample_her_transitions(buffer_temp, num_transitions) 
        obs, g = transitions['obs'], transitions['g']
        # observation을 clip한다.
        transitions['obs'], transitions['g'] = self._preproc_og(obs, g)
        #her을 통해서 transition을 대체하고 reward를 재계산한 transitions을 포함한 buffer로 normalizer를 업데이트한다.
        self.o_norm.update(transitions['obs'])
        self.g_norm.update(transitions['g'])
        #통계량 업데이트
        self.o_norm.recompute_stats()
        self.g_norm.recompute_stats()

    def _preproc_og(self, o, g):
        o = np.clip(o, -self.args.clip_obs, self.args.clip_obs)
        g = np.clip(g, -self.args.clip_obs, self.args.clip_obs)
        return o, g

    #polyak averaging을 통해 target network를 업데이트한다.
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    #네트워크를 update한다.
    def _update_network(self):
        #buffer에서 transitions을 샘플링한다.
        transitions = self.buffer.sample(self.args.batch_size)
        # observation과 goal clip 전처리.
        o, o_next, g = transitions['obs'], transitions['obs_next'], transitions['g']
        transitions['obs'], transitions['g'] = self._preproc_og(o, g)
        transitions['obs_next'], transitions['g_next'] = self._preproc_og(o_next, g)
        #normalize한다.
        obs_norm = self.o_norm.normalize(transitions['obs'])
        g_norm = self.g_norm.normalize(transitions['g'])
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        #tensor로 변환한다.
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
        # 타겟 네트워크를 활용하여 q value를 계산한다.
        with torch.no_grad():
            """
            - critic update
            
            critic을 업데이트하기 위해서 one step td update를 수행한다. 이때, 타겟값이 일정하게
            고정되게 하기 위해 타겟 네트워크를 이용한다. 그래서 타겟 넷에서 Q값을 뽑고, action도 타겟넷에서
            다음 상태를 통과시켜서 뽑은 action으로 target q value를 계산한다.
            여기서 r은 r_tensor로 리플레이버퍼로 가져오고, input next_norm tensor로 st+1을 가져온후
            actor target network로 action을 뽑아서 critic target network로 넣어서 target q value를 계산한다.
            """
            
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # td update를 위해 다음 state st+1이 아닌 st로 현재 q value를 계산한다.
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        #mse loss를 이용해 critic을 업데이트한다.
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        """
        - actor update
        트랜지션을 통해서 얻어진 샘플로 summation한 loss에 대해 critic이 예측하는 q value가 최대가 되도록 gradient ascent를 수행한다.
        => negative q-value를 minimiaze. actor로 얻어진 actions들을 critic에 넣어서 q-value를 계산하고, 이를 최대화한다.
        """
        actions_real = self.actor_network(inputs_norm_tensor)
        actor_loss = -self.critic_network(inputs_norm_tensor, actions_real).mean()
        actor_loss += self.args.action_l2 * (actions_real / self.env_params['action_max']).pow(2).mean()
        # actor network의 gradient를 구하고 update한다.
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()
        # critic network의 gradient를 구하고 update한다.
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

    # agent를 evaluation하여 success rate를 구한다.
    def _eval_agent(self):
        total_success_rate = []
        #한 에피소드에 대해서 성공률을 구한다.
        for _ in range(self.args.n_test_rollouts):
            per_success_rate = []
            observation = self.env.reset()
            
            obs = observation[0]['observation']
            g = observation[0]['desired_goal']
            for _ in range(self.env_params['max_timesteps']):
                # self.env.render()
                with torch.no_grad():
                    #학습된 actor 네트워크를 이용해 action을 선택한다.
                    input_tensor = self._preproc_inputs(obs, g)
                    pi = self.actor_network(input_tensor)
                    #action의 차원을 줄인다.
                    actions = pi.detach().cpu().numpy().squeeze()
                # print(self.env.step(actions))
                observation_new, _, _, _, info = self.env.step(actions)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                per_success_rate.append(info['is_success'])
            total_success_rate.append(per_success_rate)
        total_success_rate = np.array(total_success_rate)
        #마지막 time step에서의 성공률을 구한다.
        local_success_rate = np.mean(total_success_rate[:, -1])
        #모든 프로세스의 성공률을 더해서 평균을 구한다.
        global_success_rate = MPI.COMM_WORLD.allreduce(local_success_rate, op=MPI.SUM)
        return global_success_rate / MPI.COMM_WORLD.Get_size()
