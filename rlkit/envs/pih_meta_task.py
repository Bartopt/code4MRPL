import numpy as np
from gym import spaces
from rlkit.envs.peg_in_hole import PihEnv

from rlkit.envs import register_env


@register_env('pih-meta')
class PihMetaEnv(PihEnv):
    """
    """

    def __init__(self, task={}, n_tasks=2, randomize_tasks=True):
        self.offsetRange = 0.002
        self.conStepRange = [0.9, 1.1]
        super(PihMetaEnv, self).__init__()
        self.tasks = self.sample_tasks(n_tasks)
        self.reset_task(0)

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._task = self.tasks[idx]
        self.GoalPosOffset = self._task['GoalPosOffset']
        self.conStep = self._task['conStep']
        # goal in human demonstration, more precise than guess goal. Note that the noise should
        # identity to observation noise
        # self.demoGoalNoise = np.random.uniform(-self._obsNoiseRange, -self._obsNoiseRange, size=(3,))
        self.demoGoalNoise = np.random.uniform(-self._obsNoiseRange, self._obsNoiseRange, size=(3,))

        self.reset()

    def get_all_task_idx(self):
        return range(len(self.tasks))

    def _get_obs(self):
        return self.GetEnvObservation()

    def sample_tasks(self, num_tasks):
        # np.random.seed(1336)
        tasks = []
        for _ in range(num_tasks):
            goal_pos_offset = np.random.uniform(-self.offsetRange, self.offsetRange, size=(2,))
            controller_step = self._action_trans_bound * np.random.uniform(self.conStepRange[0], self.conStepRange[1])
            tasks.append({'GoalPosOffset': goal_pos_offset, 'conStep': controller_step})

        return tasks