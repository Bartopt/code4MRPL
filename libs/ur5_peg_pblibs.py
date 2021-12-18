import math
import os
from attrdict import AttrDict
import numpy as np
# import pybullet as p
import pybullet_data
from collections import namedtuple


class UR5:
    """
    Represents the UR5 arm in the PyBullet simulator
    :param urdf_root_path: (str) Path to pybullet urdf files
    :param timestep: (float)
    :param use_inverse_kinematics: (bool) enable dx,dy,dz control rather than direct joint control
    :param small_constraints: (bool) reduce the searchable space
    """

    def __init__(self, p, robotset):
        self.p = p

        tableStartPos = [0.0, -0.9, 0.80]
        tableStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        self.tableID = self.p.loadURDF("./libs/urdf/table.urdf", tableStartPos, tableStartOrientation,
                                       useFixedBase=True)
        # create constraint between table and box

        # ur5standStartPos = [-0, -0, 0.0]
        ur5standStartPos = [-0.7, -0.36, 0.0]
        ur5standStartOrientation = p.getQuaternionFromEuler([0, 0, 0])
        ur5standID = self.p.loadURDF("./libs/urdf/ur5_stand.urdf", ur5standStartPos, ur5standStartOrientation,
                                     useFixedBase=True)

        # define camera image parameter

        width = 224
        height = 224
        fov = 60
        aspect = width / height
        near = 0.2
        far = 2
        self.view_matrix = self.p.computeViewMatrix([0.0, -1.5, 1.5], [0, -0.5, 0.7], [0, 1, 0])
        self.projection_matrix = self.p.computeProjectionMatrixFOV(fov, aspect, near, far)
        #######################################
        ###    define and setup robot       ###
        #######################################
        self.boxStartPos = [-0.04, -0.75, 0.8265]
        self.boxStartOrientation = self.p.getQuaternionFromEuler([0, 0, 0])
        if robotset == 'ur5_peg':
            raise ValueError("make sure end effector is in the center point of the peg")
            # using cylinder will slow down the simulation during to it need more computing resource
            # self.boxId = self.p.loadURDF("./libs/urdf/insertion_box.urdf", boxStartPos, boxStartOrientation,
            #                              globalScaling=1)
            # cid = self.p.createConstraint(self.boxId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
            #                               [-0.04, -0.75, 0.82])
            # robotUrdfPath = "./libs/urdf/ur5_peg.urdf"
        elif robotset == 'ur5_cube0':
            self.boxId = self.p.loadURDF("./libs/urdf/boxcube0.urdf", self.boxStartPos, self.boxStartOrientation, globalScaling=1)
            self.boxConId = self.p.createConstraint(self.boxId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                          [-0.04, -0.75, 0.82])
            robotUrdfPath = "./libs/urdf/ur5_cube0.urdf"
        elif robotset == 'ur5_cube1':
            raise ValueError("make sure end effector is in the center point of the peg")
            # self.boxId = self.p.loadURDF("./libs/urdf/boxcube1.urdf", boxStartPos, boxStartOrientation, globalScaling=1)
            # cid = self.p.createConstraint(self.boxId, -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
            #                               [-0.04, -0.75, 0.82])
            # robotUrdfPath = "./libs/urdf/ur5_cube1.urdf"
        # setup ur5 with robotiq 85

        robotStartPos = [0, 0, 0.0]
        robotStartOrn = self.p.getQuaternionFromEuler([0, 0, 0])
        # print("----------------------------------------")
        # print("Loading robot from {}".format(robotUrdfPath))
        self.robotID = self.p.loadURDF(robotUrdfPath, robotStartPos, robotStartOrn)
        self.controlJoints = ["shoulder_pan_joint", "shoulder_lift_joint",
                              "elbow_joint", "wrist_1_joint",
                              "wrist_2_joint", "wrist_3_joint"]
        jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
        numJoints = self.p.getNumJoints(self.robotID)
        jointInfo = namedtuple("jointInfo",
                               ["id", "name", "type", "lowerLimit", "upperLimit", "maxForce", "maxVelocity",
                                "controllable"])

        self.joints = AttrDict()
        for i in range(numJoints):
            info = self.p.getJointInfo(self.robotID, i)
            jointID = info[0]
            jointName = info[1].decode("utf-8")
            jointType = jointTypeList[info[2]]
            jointLowerLimit = info[8]
            jointUpperLimit = info[9]
            jointMaxForce = info[10]
            jointMaxVelocity = info[11]
            controllable = True if jointName in self.controlJoints else False
            info = jointInfo(jointID, jointName, jointType, jointLowerLimit,
                             jointUpperLimit, jointMaxForce, jointMaxVelocity, controllable)
            # register index of dummy center link
            if jointName == "gripper_roll":
                dummy_center_indicator_link_index = i
            if info.type == "REVOLUTE":  # set revolute joint to static
                self.p.setJointMotorControl2(self.robotID, info.id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            self.joints[info.name] = info
        # joints, controlRobotiqC2, controlJoints, mimicParentName = setup_robot(p, robotID)
        p.enableJointForceTorqueSensor(self.robotID, 7, 1)
        self.eefID = 7
        self.max_velocity = .35
        self.max_force = 200.
        self.use_simulation = True
        self.use_null_space = False
        self.use_orientation = True
        # self.ur5_end_effector_index = 7
        self.jd = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
        self.controller_precision = 0.0002

        self.reset()

    def reset(self):

        """
        Reset the environment
        """
        self.setJointAngles([-1.57, -1.57, 1.57, -1.57, -1.57, 0])

    def getActionDimension(self):
        """
        Returns the action space dimensions
        :return: (int)
        """
        return 6  # position x,y,z and roll/pitch/yaw euler angles of end effector

    def getObservationDimension(self):
        """
        Returns the observation space dimensions
        :return: (int)
        """
        return len(self.getObservation())

    def getObservation(self):
        """
        Returns the position and angle of the effector
        :return: ([float])
        """
        observation = []
        state = self.p.getLinkState(self.eefID)
        pos = state[0]
        orn = state[1]
        euler = self.p.getEulerFromQuaternion(orn)

        observation.extend(list(pos))
        observation.extend(list(euler))

        return observation

    def setJointAngles(self, angles):
        for name in self.controlJoints:
            joint = self.joints[name]
            pose = angles[self.controlJoints.index(name)]
            self.p.setJointMotorControl2(self.robotID, joint.id, self.p.POSITION_CONTROL,
                                         targetPosition=pose, force=joint.maxForce,
                                         maxVelocity=joint.maxVelocity)

    def applyActions(self, x, y, z, orn):
        # calculate joint of target position
        jointPose = self.p.calculateInverseKinematics(self.robotID, self.eefID, [x, y, z], orn, jointDamping=self.jd)
        # set joint respectively
        for i, name in enumerate(self.controlJoints):
            joint = self.joints[name]
            pose = jointPose[i]

            self.p.setJointMotorControl2(self.robotID, joint.id, self.p.POSITION_CONTROL,
                                         targetPosition=pose, force=joint.maxForce,
                                         maxVelocity=joint.maxVelocity)

    def actionExplain(self, action_index):
        actions = []
        # index=0
        for x in np.arange(-0.01, 0.02, 0.01):
            for y in np.arange(-0.01, 0.02, 0.01):
                for z in np.arange(-0.01, 0.02, 0.01):
                    actions.append([x, y, z, 0])
        return actions[action_index]

    def applyRelaActions(self, translation, rotation=None, setOri=None):
        '''
        Three kinds of action: 1) only translation; 2) translation and incremental rotate; 3) traslation and rotate to
        target orientation.
        :param translation:
        :param rotation:
        :param setOri:
        :return:
        '''
        # Translation and rotation of action
        dx, dy, dz = translation
        # current position and pose of end effector
        pos, orn, _, _, _, _ = self.p.getLinkState(self.robotID, 7)
        # calculate target position and pose of end effector
        x = pos[0] + dx
        y = pos[1] + dy
        z = pos[2] + dz
        if rotation is not None:
            raise ValueError("rotate has not implement")
            # euler = self.p.getEulerFromQuaternion(orn)
            # roll = euler[0]
            # pitch = euler[1]
            # yaw = euler[2]
            # orn = self.p.getQuaternionFromEuler([roll, pitch, yaw])
        elif setOri is not None:
            orn = setOri

        # todo: if action is out of workspace, then push the peg into it
        self.applyActions(x, y, z, orn)

        # target position and orientation should not return here
        targetPosOri = np.array([x, y, z, orn[0], orn[1], orn[2], orn[3]])
        return targetPosOri

    def ReachTarget(self, targetPosOri):
        '''
        :param targetPosOri: 7 values, the first are position in Cartesian coordinate, and the last four are quaternions
        :return:
        '''
        # current position and pose of end effector
        state = self.p.getLinkState(self.robotID, 7)
        currentPosORi = np.append(np.array(state[0]), np.array(state[1]))

        # calculate the whether the position error is below the threshold
        l2Dis = np.sqrt(((currentPosORi[:3] - targetPosOri[:3]) ** 2).mean())
        if l2Dis < self.controller_precision:
            return True
        else:
            return False

    def getDistance(self):

        state = self.p.getLinkState(self.robotID, 7)
        pos = state[0]

        x = pos[0]
        y = pos[1]
        z = pos[2]

        x_t = 0
        y_t = -0.707
        z_t = 0.955

        distance = math.sqrt(math.pow(x - x_t, 2) + math.pow(y - y_t, 2 + math.pow(z - z_t, 2)))

        return x, y, z, distance

    def ResetBox(self, offset):
        # two implementation, one is reload and the other is translation
        self.p.removeConstraint(self.boxConId)
        offset = np.append(offset, 0)
        boxPos = self.boxStartPos + offset
        boxOri = self.boxStartOrientation
        self.p.resetBasePositionAndOrientation(self.boxId, boxPos, boxOri)
        self.boxConId = self.p.createConstraint(self.boxId, -1, -1, -1, self.p.JOINT_FIXED, [0, 0, 0], [0, 0, 0],
                                                [boxPos[0], boxPos[1], 0.82])
        for _ in range(10):
            self.p.stepSimulation()

    # def applyAction(self, motor_commands):
    #     """
    #     Applies the action to the effector arm
    #     :param motor_commands: (list int) dx,dy,dz,da and finger angle
    #         if inverse kinematics is enabled, otherwise 9 joint angles
    #     """
    #
    #     if self.use_inverse_kinematics:
    #
    #         dx = motor_commands[0]
    #         dy = motor_commands[1]
    #         dz = motor_commands[2]
    #         da = motor_commands[3]
    #         # finger_angle = motor_commands[4]
    #
    #         # Constrain effector position
    #         self.end_effector_pos[0] += dx
    #         self.end_effector_pos[0] = np.clip(self.end_effector_pos[0], self.min_x, self.max_x)
    #         self.end_effector_pos[1] += dy
    #         self.end_effector_pos[1] = np.clip(self.end_effector_pos[1], self.min_y, self.max_y)
    #         self.end_effector_pos[2] += dz
    #         self.end_effector_pos[2] = np.clip(self.end_effector_pos[2], self.min_z, self.max_z)
    #         self.end_effector_angle += da
    #
    #         pos = self.end_effector_pos
    #         # Fixed orientation
    #         orn = p.getQuaternionFromEuler([0, -math.pi, 0])  # -math.pi,yaw])
    #         if self.use_null_space:
    #             if self.use_orientation:
    #                 joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_index, pos, orn,
    #                                                            self.ll, self.ul, self.jr, self.rp)
    #             else:
    #                 joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_index, pos,
    #                                                            lowerLimits=self.ll, upperLimits=self.ul,
    #                                                            jointRanges=self.jr, restPoses=self.rp)
    #         else:
    #             if self.use_orientation:
    #                 joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_index, pos, orn,
    #                                                            jointDamping=self.jd)
    #             else:
    #                 joint_poses = p.calculateInverseKinematics(self.kuka_uid, self.kuka_end_effector_index, pos)
    #
    #     else:
    #         joint_poses = motor_commands
    #         self.end_effector_angle += motor_commands[7]
    #         finger_angle = motor_commands[8]
    #
    #     if self.use_simulation:
    #         # using dynamic control
    #         for i in range(self.kuka_end_effector_index + 1):
    #             p.setJointMotorControl2(bodyUniqueId=self.kuka_uid, jointIndex=i, controlMode=p.POSITION_CONTROL,
    #                                     targetPosition=joint_poses[i], targetVelocity=0, force=self.max_force,
    #                                     maxVelocity=self.max_velocity, positionGain=0.3, velocityGain=1)
    #     else:
    #         # reset the joint state (ignoring all dynamics, not recommended to use during simulation)
    #         for i in range(self.kuka_end_effector_index + 1):
    #             p.resetJointState(self.kuka_uid, i, joint_poses[i])
    #
    #     # Effectors grabbers angle
    #     p.setJointMotorControl2(self.kuka_uid, 7, p.POSITION_CONTROL, targetPosition=self.end_effector_angle,
    #                             force=self.max_force)
    #     p.setJointMotorControl2(self.kuka_uid, 8, p.POSITION_CONTROL, targetPosition=-finger_angle,
    #                             force=self.fingerA_force)
    #     p.setJointMotorControl2(self.kuka_uid, 11, p.POSITION_CONTROL, targetPosition=finger_angle,
    #                             force=self.fingerB_force)
    #
    #     p.setJointMotorControl2(self.kuka_uid, 10, p.POSITION_CONTROL, targetPosition=0,
    #                             force=self.finger_tip_force)
    #     p.setJointMotorControl2(self.kuka_uid, 13, p.POSITION_CONTROL, targetPosition=0,
    #                             force=self.finger_tip_force)
