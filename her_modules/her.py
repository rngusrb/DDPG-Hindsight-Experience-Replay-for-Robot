import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        #replay strategy: future, final, random
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k #replay_k는 일반적인 transition 중 몇개를 her transition으로 바꿀지 결정하는 변수.
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k)) #4 -> 20%, 5 -> 16.7%, 6 -> 14.3%, 7 -> 12.5%, 8 -> 11.1%, 9 -> 10%
        else:
            self.future_p = 0
        self.reward_func = reward_func #pick and place에서 reward function

    def sample_her_transitions(self, episode_batch, batch_size_in_transitions):
        T = episode_batch['actions'].shape[1] #timesteps 차원의 길이
        rollout_batch_size = episode_batch['actions'].shape[0] # batch size. 한 mpi에서 rollout_batch_size만큼의 에피소드를 수행한다.
        batch_size = batch_size_in_transitions
        # select which rollouts and which timesteps to be used
        #buffer에 저장된 배치 사이즈 중 함수의 인자인 batch_size_in_transitions만큼 랜덤하게 선택해서 어떤 rollout을 사용할 것인지를 결정한다.
        episode_idxs = np.random.randint(0, rollout_batch_size, batch_size) 
        #timesteps 중에서도 랜덤하게 선택한다.
        t_samples = np.random.randint(T, size=batch_size)
        
        #obs, ag, g, actions, obs_next, ag_next가 저장된 episode_batch에서 episode_idxs, t_samples에 해당하는 데이터를 선택한다.
        transitions = {key: episode_batch[key][episode_idxs, t_samples].copy() for key in episode_batch.keys()}
        
        
        # her idx -> relabel되는 transition의 index을 구한다. 배치사이즈에서 균등분포를 이용해 future_p만큼의 확률로 her idx를 선택한다.
        #특정 트랜지션의 goal을 achived goal로 바꾸기 위해 사용된다.
        
        her_indexes = np.where(np.random.uniform(size=batch_size) < self.future_p)
        #여기서 선택된 트랜지션은 같은 에피소드 상에서 어떤 future timestep에서 achived goal을 goal로 대체된다.
        #현 에피소드의 total timesteps에서 샘플링된 각 트랜지션의 현재 timesteps을 빼서 남아있는 timesteps를 구한다음에 random하게 그중에서 offset을 선택한다.
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        #future_offset을 더해서 future timesteps를 구한다.
        future_t = (t_samples + 1 + future_offset)[her_indexes] #남아있는 에피소드에서 랜덤하게 선택된 future timesteps, 그중에서 her_indexes에 해당하는 것만 선택된다.
        # 샘플링된 her_indexes에 해당하는 트랜지션의 goal을 future timesteps에서의 ag로 대체한다.
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        #바뀐 transition으로 reward를 다시 계산한다.
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1) #(batch_size, 1)
        #‘obs’, ‘ag’, ‘g’, ‘actions’, ‘obs_next’, ‘ag_next’, ‘r’에 대해서, reshape를 통해 (batch_size, ...)로 변환한다.
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
