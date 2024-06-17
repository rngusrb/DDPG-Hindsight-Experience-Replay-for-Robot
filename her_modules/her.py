import numpy as np

class her_sampler:
    def __init__(self, replay_strategy, replay_k, reward_func=None):
        #replay strategy: future, final, random
        self.replay_strategy = replay_strategy
        self.replay_k = replay_k
        if self.replay_strategy == 'future':
            self.future_p = 1 - (1. / (1 + replay_k))
        else:
            self.future_p = 0
        self.reward_func = reward_func

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
        future_offset = np.random.uniform(size=batch_size) * (T - t_samples)
        future_offset = future_offset.astype(int)
        future_t = (t_samples + 1 + future_offset)[her_indexes]
        # replace go with achieved goal
        future_ag = episode_batch['ag'][episode_idxs[her_indexes], future_t]
        transitions['g'][her_indexes] = future_ag
        # to get the params to re-compute reward
        transitions['r'] = np.expand_dims(self.reward_func(transitions['ag_next'], transitions['g'], None), 1)
        transitions = {k: transitions[k].reshape(batch_size, *transitions[k].shape[1:]) for k in transitions.keys()}

        return transitions
