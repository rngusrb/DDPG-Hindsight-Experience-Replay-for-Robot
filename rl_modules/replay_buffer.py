import threading
import numpy as np

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params #환경을 초기화할때 저장한다.
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T #size는 버퍼 사이즈를 타임스텝으로 나눈 값.
        # memory management
        self.current_size = 0 # 현재 저장된 데이터의 개수
        self.n_transitions_stored = 0 # 저장된 전이의 개수
        self.sample_func = sample_func
        # buffer를 만들어서 환경에서 얻어온 observation, achieved goal, goal, actions를 저장해 ddbg + her에 이용할 수 있도록 한다.
        self.buffers = {'obs': np.empty([self.size, self.T + 1, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T + 1, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T, self.env_params['action']]),
                        }
        # 동기화를 위한 락을 설정한다.
        self.lock = threading.Lock()
    
    # 에피소드를 저장한다.
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions = episode_batch
        batch_size = mb_obs.shape[0] #배치를 가져와서 저장할 때 에피소드 안에 들어있는 observation 텐서의 크기를 보고 배치 사이즈를 얻어온다.
        with self.lock: # 동기화시 상호배제를 위해 락을 건다.
            idxs = self._get_storage_idx(inc=batch_size) # 새롭게 가져온 에피소드를 저장할 때 필요한 인덱스를 가져온다.
            # key값에 해당하는 데이터를 저장한다.
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.n_transitions_stored += self.T * batch_size # 각 에피소드에 timestep T만큼의 전이가 발생하므로 총 발생한 전이의 횟수를 저장한다.
    
    #replay buffer에서 샘플링하여 학습에 이용한다. 이를 통해 데이터간의 독립성을 보장하여 데이터간의 correlation을 줄일 수 있다.
    #또한 이를 통해 데이터의 재사용을 통해 학습 효율을 높일 수 있다.
    def sample(self, batch_size):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size] #아직 리플레이 버퍼가 가득차지 않은 경우에 유효한 데이터만 사용핧 수 있도록 bound를 설정한다.
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :] #obs에서 한칸씩 땡겨서 가져와서 obs_next에 저장한다.
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        #obs_next, ag_next까지 저장한 temp_buffers를 이용해서 sample_func을 이용해서 샘플링을 한다.
        transitions = self.sample_func(temp_buffers, batch_size)
        return transitions
    #새로운 데이터를 저장할 때 필요한 인덱스를 가져온다.
    
    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:# 버퍼가 가득차기 전에는 순서대로 저장될 인덱스를 리턴한다.
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size: #inc 만큼 데이터를 증가시켰을 때 일부분이 버퍼를 넘어가는 경우.
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)#넘지 않은데 까지는 그대로 인덱스로 사용한다.
            idx_b = np.random.randint(0, self.current_size, overflow) #넘으면 랜덤하게 기존의 데이터를 덮어쓰기할 인덱스를 가져온다.
            idx = np.concatenate([idx_a, idx_b]) #이를 합쳐서 인덱스로 사용한다.
        else:
            idx = np.random.randint(0, self.size, inc) #버퍼가 가득차있는 경우에는 랜덤하게 인덱스를 가져온다.
        self.current_size = min(self.size, self.current_size+inc) #더 늘어나거나 아니면 덮어쓰기 해서 사이즈가 그대로거나.
        if inc == 1:
            idx = idx[0]
        return idx
