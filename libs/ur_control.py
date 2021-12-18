import os
import time
import pdb
import pybullet as p
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import functools


def robotSetJointAngles(p, robotID, controlJoints, joints, angles, vel):
    for name in controlJoints:
        joint = joints[name]
        pose = angles[controlJoints.index(name)]
        p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,
                                targetPosition=pose, force=joint.maxForce,
                                maxVelocity=joint.maxVelocity)


def robotCarresianControl(p, robotID, eefID, joints, controlJoints, jd, x, y, z, roll, pitch, yaw, vel):
    orn = p.getQuaternionFromEuler([roll, pitch, yaw])

    # apply IK
    jointPose = p.calculateInverseKinematics(robotID, eefID, [x, y, z], orn, jointDamping=jd)
    # print(jointPose)
    for i, name in enumerate(controlJoints):
        joint = joints[name]
        pose = jointPose[i]

        p.setJointMotorControl2(robotID, joint.id, p.POSITION_CONTROL,
                                targetPosition=pose, force=joint.maxForce,
                                maxVelocity=joint.maxVelocity * vel)
