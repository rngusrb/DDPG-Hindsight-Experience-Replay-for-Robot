from mpi4py import MPI
import numpy as np
import torch

#여러 프로세서에서 돌아가는 네트워크를 동기화시킴. -> 주 프로세서에서 네트워크의 파라미터를 가져와서 모든 프로세서에게 브로드캐스팅.
def sync_networks(network):
    """
    netowrk is the network you want to sync

    """
    comm = MPI.COMM_WORLD
    flat_params = _get_flat_params_or_grads(network, mode='params') #네트워크의 파라미터를 플랫하게 가져옴.
    comm.Bcast(flat_params, root=0) #주 프로세서에서 모든 프로세서에게 브로드캐스팅.
    # set the flat params back to the network
    _set_flat_params_or_grads(network, flat_params, mode='params') #네트워크에 파라미터를 설정함.
    
    
#여러 프로세서에서 네트워크의 그래디언트를 동기화시킴. 각 프로세서의 그래디언트를 모두 합산하고, 그 결과를 모든 프로세서에게 브로드캐스팅.
def sync_grads(network):
    flat_grads = _get_flat_params_or_grads(network, mode='grads') #네트워크의 그래디언트를 플랫하게 가져옴.
    comm = MPI.COMM_WORLD #MPI.COMM_WORLD는 모든 프로세서를 포함하는 커뮤니케이터.
    global_grads = np.zeros_like(flat_grads) #모든 프로세서의 그래디언트를 합칠 변수를 flat_grads의 차원만큼 영행렬로 만듦.
    comm.Allreduce(flat_grads, global_grads, op=MPI.SUM) #모든 프로세서의 그래디언트를 합산함.
    _set_flat_params_or_grads(network, global_grads, mode='grads') #합산된 그래디언트를 네트워크에 설정함.

# 네트워크의 파라미터나 그래디언트를 플랫 배열로 변환하는 함수.
def _get_flat_params_or_grads(network, mode='params'):
    
    attr = 'data' if mode == 'params' else 'grad'
    return np.concatenate([getattr(param, attr).cpu().numpy().flatten() for param in network.parameters()])
#네트워크의 파라미터나 그래디언트를 플랫 배열에서 네트워크의 원래 구조로 복원하는 역할.
def _set_flat_params_or_grads(network, flat_params, mode='params'):
    """
    include two kinds: grads and params

    """
    attr = 'data' if mode == 'params' else 'grad'
    # the pointer
    pointer = 0
    for param in network.parameters(): #포인터로 범위를 설정하고, 네트워크의 파라미터나 그래디언트를 플랫 배열에서 네트워크의 원래 구조로 복원함.
        getattr(param, attr).copy_(torch.tensor(flat_params[pointer:pointer + param.data.numel()]).view_as(param.data))
        pointer += param.data.numel()
