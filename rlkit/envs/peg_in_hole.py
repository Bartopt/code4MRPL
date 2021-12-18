import os
from collections import deque
import pybullet as p
import pybullet_data
import numpy as np
from gym import spaces
import torch as th
from libs.ur5_peg_pblibs import *
from libs.base_env import BaseGymEnv
# from state_representation.episode_saver import EpisodeSaver

# DELTA_V = 0.01
# NOISE_STD = 0.002
# DELTA_V_CONTINUOUS = 0.01
# NOISE_STD_CONTINUOUS = 0.0001
N_CONTACTS_BEFORE_TERMINATION = 5
# Terminate the episode if the arm is outside the safety sphere during too much time
N_STEPS_OUTSIDE_SAFETY_SPHERE = 5000


class PihEnv(BaseGymEnv):
    def __init__(self, robot_set='ur5_cube0', rewardFalg='naive', Resi=True, force_down=False, renders=True,
                 is_discrete=False, action_joints=False, renderSwitch=False, debug=False, onlyRes=False,
                 useDemo=False, randomSearch=False, **_):
        '''

        Args:
            robot_set:
            rewardFalg:
            Resi:
            force_down:
            renders:
            is_discrete:
            action_joints:
            renderSwitch:
            debug:
            onlyRes: only use controller for comparing, set true when evaluate C1, C2
            useDemo: use goal in demo to update controller's input. set true when evaluate C2 and MRPL.
            **_:

            Q2
            MRPL: model:MRPL, demo(sim_policy.py):1, resi:T, onlyRes:F, useDemo:T
            PEARL: model:PEARL, demo:0, resi:F, onlyRes:F, useDemo:F
            RandomSearch: model:None, demo:0, resi:T, onlyRes:T, useDemo:F, randomSearch: T

            Q4
            MRPL: model:MRPL, demo:1, resi:T, onlyRes:F, useDemo:T
            MRPL-NORes: model:PEARL, demo:1, resi:F, onlyRes:F, useDemo:T
            MRPL-NODemo: model:MRPL, demo:0, resi:T, onlyRes:F, useDemo:F
            MRPL-NORes-NODemo: model:PEARL, demo:0, resi:F, onlyRes:F, useDemo:F
            GuessController: model:None, demo:0, resi:T, onlyRes:T, useDemo:F
            DemoController: model:None, demo:0, resi:T, onlyRes:T, useDemo:T
        '''

        # parameters
        self.Resi = Resi
        self.debug = False
        self.n_contacts = 0
        self.action_joints = action_joints
        self.cuda = th.cuda.is_available()
        self.saver = None
        self.robot = None
        self.action = None
        self.workSpace = None
        self.targetPos = np.array([0, -0.71, 0.8815])
        self.startEEPos = [0, -0.71, 0.9345]
        self.startEEOri = None
        self.GoalPosOffset = None
        self.conStep = None
        self._rewardFlag = rewardFalg
        self._timestep = 1. / 240
        self._robot_set = robot_set
        self._observation = []
        self._env_step_counter = 0
        self._render = renders
        self._width = 224
        self._height = 224
        self._force_down = force_down
        self._is_discrete = is_discrete
        self._action_trans_bound = 0.002
        self._maxReachAttempts = 20
        self._terminatedTS = 0.005
        self._renderSwitch = renderSwitch
        self._insectTS = -0.03
        self._maxReward = -999
        self._obsNoiseRange = 0.001
        self._debug = debug
        self.useDemo = useDemo
        self.onlyRes = onlyRes
        self.randomSearch = randomSearch

        # Visualization or Not
        if self._render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        self.LoadScene()
        # define action_space
        # 定义一个Discrete类的空间只需要一个参数n就可以了，而定义一个多维的Box空间需要知道每一个维度的最小最大值,and维数
        if self._is_discrete:
            # guess six actions represent up, down, left, right, forward and back
            self.action_space = spaces.Discrete(6)
        else:
            if self.action_joints:
                # continuous action include transition and rotation
                action_dim = 6
            else:
                # continuous action include transition without rotation
                action_dim = 3
            action_high = np.array([self._action_trans_bound] * action_dim)
            self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)
        # define observation_space
        # observations include relative x, y, z, the range does not limit the observations
        self.observation_space = spaces.Box(-np.inf, np.inf, shape=(3,), dtype=np.float32)

        # self.GetWorkSpace()

    # def GetWorkSpace(self):
    #     '''
    #     work space is defined with a cuboid, where the end effector can not move out. centerPoint is the center of the
    #     workspace cuboid.
    #     :return:
    #     '''
    #     cuboidSize = np.array([0.07, 0.07, 0.22]) # length, width, height
    #     centerPoint = self.getTargetPos()
    #     upperBound = centerPoint + cuboidSize / 2
    #     lowerBound = centerPoint - cuboidSize / 2
    #     self.workSpace = np.vstack((lowerBound, upperBound))

    def getTargetPos(self):
        '''
        End effector point is in the center of the peg upper surface, so target position is the center of the hole upper
        surface. The coordinate origin of the box is on the corner of bottom surface, so
        targetPos = boxPos +/- [length/2, length/2, height]
        :return:
        '''
        return self.targetPos

    def getTargetOrn(self):
        return np.array([-1.57])

    def getPegOrn(self):
        return np.array(p.getEulerFromQuaternion(p.getLinkState(self.robot.robotID, self.robot.eefID)[1])[2])

    def getPegPos(self):
        """
        :return: ([float]) Position (x, y, z) of kuka gripper
        """
        return np.array(p.getLinkState(self.robot.robotID, self.robot.eefID)[0][:3])

    def getWristForce(self):
        return np.array(p.getJointState(self.robot.robotID, self.robot.eefID)[2])

    def LoadScene(self):
        p.resetSimulation()
        p.setPhysicsEngineParameter(numSolverIterations=150)
        p.setTimeStep(self._timestep)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        # define world
        p.setGravity(0, 0, -9.8)  # NOTE
        planeID = p.loadURDF("plane.urdf")
        if self._robot_set == 'ur5_peg':
            self.robot = UR5(p, self._robot_set)
        elif self._robot_set == 'ur5_cube0':
            self.robot = UR5(p, self._robot_set)
        elif self._robot_set == 'ur5_cube1':
            self.robot = UR5(p, self._robot_set)
        self.robot.setJointAngles([-1.57, -1.57, 1.57, -1.57, -1.57, 0])
        for _ in range(500):
            p.stepSimulation()

        state = p.getLinkState(self.robot.robotID, 7)
        self.startEEOri = state[1]

    def RenderSwitch(self, render):
        p.disconnect()
        self._render = render
        if render:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        self.LoadScene()

    def reset(self):
        self._obsNoise = np.random.uniform(-self._obsNoiseRange, -self._obsNoiseRange, size=(3,))
        if self._debug:
            print('reset the env')
        # render the env when reward is over the threshold
        if self._renderSwitch:
            if self._maxReward > self._insectTS and self._render is False:
                self.RenderSwitch(render=True)
                print('Insert successfully, start visualization training')

        # to avoid reload too much times, move to auxiliary position first.
        auxPos = self.getPegPos()
        auxPos[2] = 0.92
        targetPosOri = np.append(auxPos, self.startEEOri)
        curAttempts = 0
        self.robot.applyActions(auxPos[0], auxPos[1], auxPos[2], self.startEEOri)
        while not self.robot.ReachTarget(targetPosOri):
            p.stepSimulation()
            curAttempts += 1
            # reload all object if attempts over threshold
            if curAttempts > 100:
                break

        # generate start pos with offset
        targetPosOri = np.append(self.startEEPos, self.startEEOri)
        # move robot to start pos
        curAttempts = 0
        self.robot.applyActions(self.startEEPos[0], self.startEEPos[1], self.startEEPos[2], self.startEEOri)
        while not self.robot.ReachTarget(targetPosOri):
            p.stepSimulation()
            curAttempts += 1
            # reload all object if attempts over threshold
            if curAttempts > 200:
                self.LoadScene()
                curAttempts = 0
                self.robot.applyActions(self.startEEPos[0], self.startEEPos[1], self.startEEPos[2], self.startEEOri)

        # reset box and control step according tasks
        self.robot.ResetBox(self.GoalPosOffset)
        if self.action_joints:
            # continuous action include transition and rotation
            action_dim = 6
        else:
            # continuous action include transition without rotation
            action_dim = 3
        action_high = np.array([self.conStep] * action_dim)
        self.action_space = spaces.Box(-action_high, action_high, dtype=np.float32)

        # get observation
        self._observation = self.GetEnvObservation()
        self._env_step_counter = 0
        self.rSCount = 0

        return np.array(self._observation)

    def __del__(self):
        p.disconnect()

    def GetEnvObservation(self, GetForce=False):
        position = np.array(self.getTargetPos() - self.getPegPos())
        if GetForce:
            force = np.array(self.getWristForce()[2])
            return np.append(position, force)
        else:
            return np.copy(position + self._obsNoise)

    def step(self, action):
        if action is None:
            self.robot.applyRelaActions([0, 0, 0])

        if self._is_discrete:
            raise ValueError("discrete action does not implementation")
        else:
            dx = action[0]
            dy = action[1]
            # todo: add noise on action
            if self._force_down:
                dz = -abs(action[2])  # Remove up action
            else:
                dz = action[2]

            real_action = [dx, dy, dz]

            if self.Resi == True:
                # this is goal without noise
                # offset = np.append(self.GoalPosOffset, 0)
                # CurTargetPos = IniTargetPos + offset
                # tx, ty, tz = CurTargetPos
                if self.useDemo:
                    demoGoal = self.GetEnvObservation() + np.append(self.GoalPosOffset, 0) + self.demoGoalNoise
                    px, py, pz = 0.2 * demoGoal
                else:
                    if self.randomSearch:
                        if self.rSCount == 0:
                            self.offX, self.offY = np.random.uniform(-0.0015, 0.0015, size=(2,))

                        px, py, pz = 0.2 * self.GetEnvObservation()
                        px += self.offX
                        py += self.offY
                        self.rSCount += 1

                        # if attempt 5 times and still failed to insert, then change guessed position
                        if self.rSCount > 5 and self.GetEnvObservation()[2] < -0.05:
                            self.rSCount = 0

                    else:
                        px, py, pz = 0.2 * self.GetEnvObservation()

                if self._env_step_counter > 3:
                    pz = 0.2 * pz
                else:
                    pz = 0
                if self.onlyRes:
                    real_action = [px, py, pz]
                else:
                    real_action = [dx + px, dy + py, dz + pz]

        targetPosOri = self.robot.applyRelaActions(real_action, setOri=self.startEEOri)
        curAttempts = 0
        while not self.robot.ReachTarget(targetPosOri):
            p.stepSimulation()
            curAttempts += 1
            if curAttempts > self._maxReachAttempts:
                break

        self._env_step_counter += 1
        self._observation = self.GetEnvObservation()

        reward = self._reward()
        # done, doneInfo = self._termination()
        done = False

        info = {}
        info['IsSuc'] = self.IsSuc(reward)
        # if done:
        #     if doneInfo == 'insert successful':
        #         info['is_success'] = True
        #     else:
        #         info['is_success'] = False
        #     if self._maxReward < reward:
        #         self._maxReward = reward
        #     if self._debug:
        #         print('this epoch has done, done information: {0}, this epoch have {1} timesteps'.format(doneInfo, str(self._env_step_counter)))

        return np.array(self._observation), reward, done, info

    def IsSuc(self, reward):
        if reward > -self._terminatedTS:
            return 1.
        else:
            return 0.

    # def IsOutOfSpace(self):
    #     eePos = self.getPegPos()
    #     if np.all((eePos - self.workSpace[0]) > 0) and np.all((eePos - self.workSpace[1]) < 0):
    #         return False
    #     else:
    #         return True

    # def _termination(self):
    #     if self._env_step_counter > self.episode_max_steps:
    #         self._observation = self.GetEnvObservation()
    #         return True, 'reaching max steps'
    #     # if np.abs(self.GetEnvObservation()[2]) < self._terminatedTS:
    #     #     self._observation = self.GetEnvObservation()
    #     #     return True, 'insert successful'
    #     # if self.IsOutOfSpace():
    #     #     self._observation = self.GetEnvObservation()
    #     #     return True, 'out of work space'
    #     return False, 'NOT TERMINATE'

    def _naivereward(self):
        '''
        The naive reward is L2 distance. If the distance large or the peg is contact with box, the set reward as -2.
        :return:
        '''
        IniTargetPos = self.getTargetPos()
        offset = np.append(self.GoalPosOffset, 0)
        CurTargetPos = IniTargetPos + offset
        distance = np.linalg.norm(CurTargetPos - self.getPegPos(), 2)
        # print(distance)
        reward = -distance
        # contact_with_table = len(p.getContactPoints(self.robot.boxId, self.robot.robotID)) > 0
        # if distance > self._max_distance or contact_with_table:
        #     reward = -2

        # if contact_with_table or self.n_contacts >= N_CONTACTS_BEFORE_TERMINATION \
        #         or self.n_steps_outside >= N_STEPS_OUTSIDE_SAFETY_SPHERE:
        #     self.terminated = True
        # if distance < 0.01:
        #     # print(distance,self.getPegPos(),self.getTargetPos())
        #     reward = 10
        return reward

    def _sparsereward(self):
        threshold = 0.005
        cur_heigh = abs(np.array(self.getTargetPos() - self.getPegPos())[2])

        if cur_heigh > threshold:
            reward = -0.05
        else:
            reward = 0

        return reward

    def _reward(self):
        if self._rewardFlag == "naive":
            return self._naivereward()
        elif self._rewardFlag == "shape":
            return self._shapereward()
        elif self._rewardFlag == "sparse":
            return self._sparsereward()

    def SampleStartPos(self, range):
        x = self.startEEPos[0] + np.random.uniform(-0.5, 0.5) * range
        y = self.startEEPos[1] + np.random.uniform(-0.5, 0.5) * range
        z = self.startEEPos[2] + np.random.uniform(0, 1) * range

        return [x, y, z]

