import numpy as np
from gym import spaces
from gym import Env

from . import register_env


@register_env('point-robot')
class PointEnv(Env):
    """
    point robot on a 2-D plane with position control
    tasks (aka goals) are positions on the plane

     - tasks sampled from unit square
     - reward is L2 distance
    """

    def __init__(self, randomize_tasks=False, n_tasks=2, resi=True, controlBelief=False, onlyRes=False, useDemo=True):
        '''
        onlyRes: only use controller for comparing, set true when evaluate C1, C2
        useDemo: use goal in demo to update controller's input. set true when evaluate C2 and MRPL.

        Q2
        MRPL: model:MRPL, demo:1, resi:T, onlyRes:F, useDemo:T
        PEARL: model:PEARL, demo:0, resi:F, onlyRes:F, useDemo:F

        Q4
        MRPL: model:MRPL, demo:1, resi:T, onlyRes:F, useDemo:T
        MRPL-NORes: model:PEARL, demo:1, resi:F, onlyRes:F, useDemo:T
        MRPL-NODemo: model:MRPL, demo:0, resi:T, onlyRes:F, useDemo:F
        MRPL-NORes-NODemo: model:PEARL, demo:0, resi:F, onlyRes:F, useDemo:F
        GuessController: model:None, demo:0, resi:T, onlyRes:T, useDemo:F
        DemoController: model:None, demo:0, resi:T, onlyRes:T, useDemo:T

        '''
        if randomize_tasks:
            np.random.seed(1337)
            goals = [[np.random.uniform(-1., 1.), np.random.uniform(-1., 1.)] for _ in range(n_tasks)]
        else:
            # some hand-coded goals for debugging
            goals = [np.array([10, -10]),
                     np.array([10, 10]),
                     np.array([-10, 10]),
                     np.array([-10, -10]),
                     np.array([0, 0]),

                     np.array([7, 2]),
                     np.array([0, 4]),
                     np.array([-6, 9])
                     ]
            goals = [g / 10. for g in goals]
        self.goals = goals
        self.resi = resi
        self.useDemo = useDemo
        self.onlyRes = onlyRes
        self.conBelief = controlBelief

        self.reset_task(0)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(2,))
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,))

    def GuessGoals(self):
        ''' guess a goal as input of p controller '''
        self.guessGoals = np.array(self._goal) + np.random.uniform(-0.4, 0.4, size=(2,))

    def DemoGoal(self):
        '''goal in human demonstration, more precise than guess goal. Note that the noise should
        identity to observation noise'''
        self.demoGoal = np.array(self._goal) + np.random.uniform(-0.2, 0.2, size=(2,))

    def reset_task(self, idx):
        ''' reset goal AND reset the agent '''
        self._goal = self.goals[idx]
        self.GuessGoals()
        self.DemoGoal()
        self.reset()

    def get_all_task_idx(self):
        return range(len(self.goals))

    def reset_model(self):
        # reset to a random location on the unit square
        self.controlBelief = 1.0
        self.rewardBuffer = []
        self._state = np.random.uniform(-1., 1., size=(2,))
        self._obsNoise = np.random.uniform(-0.2, 0.2, size=(2,))
        return self._get_obs()

    def reset(self):
        return self.reset_model()

    def _get_obs(self):
        return np.copy(self._state + self._obsNoise)

    def UpdateControlBelief(self, pAction, cAction):
        if len(self.rewardBuffer) < 2:
            return

        factor = [0.5, 2]
        action = pAction + cAction

        angle = np.arccos(action.dot(cAction) / (np.sqrt(action.dot(action)) * np.sqrt(cAction.dot(cAction)))) * 180 / np.pi

        # if reward increase and the moving direction of cAction is same with action, the increase the belief
        if self.rewardBuffer[-1] > self.rewardBuffer[-2] and angle < 60:
            self.controlBelief = max(1.5, self.controlBelief * factor[1])
        elif self.rewardBuffer[-1] < self.rewardBuffer[-2] and angle > 120:
            self.controlBelief = max(1.5, self.controlBelief * factor[1])
        else:
            self.controlBelief = min(0.1, self.controlBelief * factor[0])

    def step(self, pAction):
        if self.resi:
            p = 0.2
            if self.useDemo:
                cAction = p * (self.demoGoal - self._get_obs()) * self.controlBelief
            else:
                cAction = p * (self.guessGoals - self._get_obs()) * self.controlBelief
            if self.onlyRes:
                self._state = self._state + cAction
            else:
                self._state = self._state + cAction + pAction
        else:
            self._state = self._state + pAction
        x, y = self._state
        x -= self._goal[0]
        y -= self._goal[1]
        reward = - (x ** 2 + y ** 2) ** 0.5
        done = False
        ob = self._get_obs()

        # updata controller belief
        if self.conBelief:
            self.rewardBuffer.append(reward)
            self.UpdateControlBelief(pAction, cAction)

        return ob, reward, done, dict()

    def viewer_setup(self):
        print('no viewer')
        pass

    def render(self):
        print('current state:', self._state)


@register_env('sparse-point-robot')
class SparsePointEnv(PointEnv):
    '''
     - tasks sampled from unit half-circle
     - reward is L2 distance given only within goal radius

     NOTE that `step()` returns the dense reward because this is used during meta-training
     the algorithm should call `sparsify_rewards()` to get the sparse rewards
     '''
    def __init__(self, randomize_tasks=False, n_tasks=2, goal_radius=0.2):
        super().__init__(randomize_tasks, n_tasks)
        self.goal_radius = goal_radius

        if randomize_tasks:
            np.random.seed(1337)
            radius = 1.0
            angles = np.linspace(0, np.pi, num=n_tasks)
            xs = radius * np.cos(angles)
            ys = radius * np.sin(angles)
            goals = np.stack([xs, ys], axis=1)
            np.random.shuffle(goals)
            goals = goals.tolist()

        self.goals = goals
        self.reset_task(0)

    def sparsify_rewards(self, r):
        ''' zero out rewards when outside the goal radius '''
        mask = (r >= -self.goal_radius).astype(np.float32)
        r = r * mask
        return r

    def reset_model(self):
        self.controlBelief = 1.0
        self.rewardBuffer = []
        self._state = np.array([0, 0])
        self._obsNoise = np.random.uniform(-0.2, 0.2, size=(2,))
        return self._get_obs()

    def step(self, action):
        ob, reward, done, d = super().step(action)
        sparse_reward = self.sparsify_rewards(reward)
        # make sparse rewards positive
        if reward >= -self.goal_radius:
            sparse_reward += 1
        d.update({'sparse_reward': sparse_reward})
        return ob, reward, done, d
