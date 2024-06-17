import threading
import numpy as np
from mpi4py import MPI

class normalizer:
    def __init__(self, size, eps=1e-2, default_clip_range=np.inf):
        self.size = size
        self.eps = eps
        self.default_clip_range = default_clip_range
        # some local information
        self.local_sum = np.zeros(self.size, np.float32)
        self.local_sumsq = np.zeros(self.size, np.float32)
        self.local_count = np.zeros(1, np.float32)
        # get the total sum sumsq and sum count
        self.total_sum = np.zeros(self.size, np.float32)
        self.total_sumsq = np.zeros(self.size, np.float32)
        self.total_count = np.ones(1, np.float32)
        # get the mean and std
        self.mean = np.zeros(self.size, np.float32)
        self.std = np.ones(self.size, np.float32)
        # thread locker
        self.lock = threading.Lock()
    
    #새로운 값을 받아서 업데이트.
    def update(self, v):
        v = v.reshape(-1, self.size)
        # 로컬 값들을 v로 업데이트함.
        with self.lock:
            self.local_sum += v.sum(axis=0)
            self.local_sumsq += (np.square(v)).sum(axis=0)
            self.local_count[0] += v.shape[0]

    #각각 로컬로 저장된 통계를 모든 프로세서에게 동기화시킴.
    def sync(self, local_sum, local_sumsq, local_count):
        local_sum[...] = self._mpi_average(local_sum)
        local_sumsq[...] = self._mpi_average(local_sumsq)
        local_count[...] = self._mpi_average(local_count)
        return local_sum, local_sumsq, local_count
    #동기화된 통계값을 기반으로 새로운 평균과 표준편차를 계산한다.
    def recompute_stats(self):
    
        with self.lock:
            local_count = self.local_count.copy()
            local_sum = self.local_sum.copy()
            local_sumsq = self.local_sumsq.copy()
            #초기화
            self.local_count[...] = 0
            self.local_sum[...] = 0
            self.local_sumsq[...] = 0
        # 통계값들을 동기화시키고 동기화된 값들을 가져온다.
        sync_sum, sync_sumsq, sync_count = self.sync(local_sum, local_sumsq, local_count)
        # update the total stuff
        self.total_sum += sync_sum
        self.total_sumsq += sync_sumsq
        self.total_count += sync_count
        #새롭게 얻은 통계값을 기반으로 평균을 다시 계산함.
        self.mean = self.total_sum / self.total_count
        # 표준편차를 다시 계산하기 위해서 E(x^2) - (Ex)^2를 계산함.
        self.std = np.sqrt(np.maximum(np.square(self.eps), (self.total_sumsq / self.total_count) - np.square(self.total_sum / self.total_count)))
    
    # 모든 프로세서의 값을 합산한 후, 모든 프로세서의 개수로 나누어 평균을 계산함.
    def _mpi_average(self, x):
        buf = np.zeros_like(x)
        MPI.COMM_WORLD.Allreduce(x, buf, op=MPI.SUM) #모든 프로세서의 x를 합산한 값을 buf에 저장함.
        buf /= MPI.COMM_WORLD.Get_size() #모든 프로세서의 개수로 나눔.
        return buf

    # 정규화 및 클립핑을 진행함.
    def normalize(self, v, clip_range=None):
        if clip_range is None:
            clip_range = self.default_clip_range
        return np.clip((v - self.mean) / (self.std), -clip_range, clip_range)
