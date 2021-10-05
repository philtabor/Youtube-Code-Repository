import torch.multiprocessing as mp
from actor_critic import ActorCritic
from icm import ICM
from shared_adam import SharedAdam
from worker import worker


class ParallelEnv:
    def __init__(self, env_id, input_shape, n_actions, icm, n_threads=8):
        names = [str(i) for i in range(1, n_threads+1)]

        global_actor_critic = ActorCritic(input_shape, n_actions)
        global_actor_critic.share_memory()
        global_optim = SharedAdam(global_actor_critic.parameters())

        if not icm:
            global_icm = None
            global_icm_optim = None
        else:
            global_icm = ICM(input_shape, n_actions)
            global_icm.share_memory()
            global_icm_optim = SharedAdam(global_icm.parameters())

        self.ps = [mp.Process(target=worker,
                              args=(name, input_shape, n_actions,
                                    global_actor_critic, global_icm,
                                    global_optim, global_icm_optim, env_id,
                                    n_threads, icm))
                   for name in names]

        [p.start() for p in self.ps]
        [p.join() for p in self.ps]
