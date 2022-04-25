import numpy as np
import time
from robot_interface import RobotInterface 


def prepare_position_cmd(target_joint_position, Kp, Kd):
    '''
        Prepare low level command according to joint target position 
    
    Input:
        target_joint_position - numpy array of shape (12), contains target joint angle in radians.
    '''
    cmd = np.zeros(60, dtype=np.float32)
    for i in range(12):
        cmd[i*5] = target_joint_position[i]
        cmd[i*5+1] = Kp
        cmd[i*5+2] = 0
        cmd[i*5+3] = Kd
        cmd[i*5+4] = 0
    return cmd


def prepare_high_level_cmd(high_action):
    '''
        Prepare low level command according to joint target position 
    
    Input:
        target_joint_position - numpy array of shape (12), contains target joint angle in radians.
    '''
    #cmd = np.zeros(60, dtype=np.float32)
    forward_speed = np.clip(high_action[0], -0.05, 0.3)
    rotate_speed = high_action[1]

    high_action = np.array([2.0, forward_speed, rotate_speed])
    return high_action


def interpolate_joint_position(pos_1, pos_2, p):
    '''
    Interpolate between joint position 1 and joint position 2

    Input:
        pos_1 - numpy array of shape (12), contains initial joint position in radians.
        pos_2 - numpy array of shape (12), contains end joint position in radians.
        p - interpolate coefficient, a number in [0.0,1.0]. 
            0.0 represents initial position, 1.0 represents end position.
            number in between will output linear combination of the positions.

    Output:
        numpy array of shape (12), interpolated joint angles. 
    '''

    # check size
    assert pos_1.shape == (12,)
    assert pos_2.shape == (12,)

    # constrain p between 0.0 and 1.0
    p = min(1.0,p)
    p = max(0.0,p)

    ## TODO cautious!!! Do we need to worry about angle crossover? 
    return (1.0 - p) * pos_1 + p * pos_2


def observation_to_joint_position(observation):
    '''
    Extract joint position information from observation

    Input: 
        observation - observation by calling receive_observation()

    Output:
        numpy array of shape (12), all joint angles, in radians. 
    '''

    joint_position = np.zeros(12, dtype=np.float32)

    for i in range(12):
        joint_position[i] = observation.motorState[i].q

    return joint_position


def observation_to_torque(observation):
    '''
    Extract torque information from observation

    Input: 
        observation - observation by calling receive_observation()

    Output:
        numpy array of shape (12), all torque
    '''

    torque = np.zeros(12, dtype=np.float32)

    for i in range(12):
        torque[i] = observation.motorState[i].tau

    return torque


def observation_to_joint_state(observation):
    '''
    Extract joint position, velocity, torque from observation

    Input: 
        observation - observation by calling receive_observation()

    Output:
        numpy array of shape (36), [all joint angles,all joint velocity, all torque] 
    '''

    joint_state = np.zeros(36, dtype=np.float32)

    for i in range(12):
        joint_state[i+0] = observation.motorState[i].q
        joint_state[i+12] = observation.motorState[i].dq
        joint_state[i+24] = observation.motorState[i].tauEst

    return joint_state


def check_joint_angle_sanity(joint_position):
    '''
    Check if the joint angle reported by the robot is correct. 

    This will check:
        1. is the returned joint position within the limit?
        2. Are they not all zeros? (all zeros probably indicates no return data)

    Input:
        joint_position - numpy array of shape (12,). 

    Output:
        True if passed check. 
    '''

    A1_joint_angle_limits = np.array([
        [-0.802,    -1.05,      -2.73],  # Hip, Thigh, Calf Min
        [0.802,     4.19,       -0.916] # Hip, Thigh, Calf Max
    ])

    # check input shape
    assert joint_position.shape == (12,)

    # check is angles within limit
    is_within_limit = np.logical_and(
        joint_position.reshape(4,3) >= A1_joint_angle_limits[0,:],
        joint_position.reshape(4,3) <= A1_joint_angle_limits[1,:]
    ).all()

    # check is angles not all zero
    # already checked by above
    return is_within_limit

