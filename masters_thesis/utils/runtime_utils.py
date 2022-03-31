import numpy as np
import math
from abr_control.utils import transformations as transform

def convert_angles(ang):
    """ Converts Euler angles from x-y-z to z-x-y convention """

    def b(num):
        """ forces magnitude to be 1 or less """
        if abs( num ) > 1.0:
            return math.copysign( 1.0, num )
        else:
            return num

    s1 = math.sin(ang[0])
    s2 = math.sin(ang[1])
    s3 = math.sin(ang[2])
    c1 = math.cos(ang[0])
    c2 = math.cos(ang[1])
    c3 = math.cos(ang[2])

    pitch = math.asin(b(c1*c3*s2-s1*s3) )
    cp = math.cos(pitch)
    # just in case
    if cp == 0:
        cp = 0.000001

    yaw = math.asin(b((c1*s3+c3*s1*s2)/cp) ) #flipped
    # Fix for getting the quadrants right
    if c3 < 0 and yaw > 0:
        yaw = math.pi - yaw
    elif c3 < 0 and yaw < 0:
        yaw = -math.pi - yaw

    roll = math.asin(b((c3*s1+c1*s2*s3)/cp) ) #flipped
    return [roll, pitch, yaw]


def ego_state_error(x):
    """
    Takes in 12D state and 12D target, returns 12D egocentric state error
    Parameters
    ----------
    x : (n, 24) array of floats
        [
            x, y, z
            dx, dy, dz,
            a, b, g,
            da, db, dg,
            target_x, target_y, target_z,
            target_dx, target_dy, target_dz,
            target_a, target_b, target_g,
            target_da, target_db, target_dg
        ]
    """
    # Find the error
    # convert our target alpha-beta targets from world to drone space
    # get our target global rotation as a quaternion
    target_quat = transform.quaternion_from_euler(x[18], x[19], 0, 'rxyz')

    # get our current orientation as a quaternion
    rot_matrix = np.array([
        [np.cos(x[8]), -np.sin(x[8]), 0],
        [np.sin(x[8]), np.cos(x[8]), 0],
        [0, 0, 1]
    ])

    quat_rotation = transform.quaternion_from_matrix(rot_matrix)

    # rotate our target quat by yaw to get our rotation of our drone
    final_quat = transform.quaternion_multiply(target_quat, quat_rotation)

    # convert to euler angles
    final_euler = list(transform.euler_from_quaternion(final_quat, 'rxyz'))

    final_euler = convert_angles(final_euler)

    x[18] = final_euler[0]
    x[19] = final_euler[1]


    ori_err = [x[18] - x[6],
                x[19] - x[7],
                x[20] - x[8]]

    for ii in range(3):
        if ori_err[ii] > math.pi:
            ori_err[ii] -= 2 * math.pi
        elif ori_err[ii] < -math.pi:
            ori_err[ii] += 2 * math.pi


    cz = math.cos(x[8])
    sz = math.sin(x[8])

    x_err = x[12] - x[0]
    y_err = x[13] - x[1]
    pos_err = [
        x_err * cz + y_err * sz,
        -x_err * sz + y_err * cz,
        x[14] - x[2]
    ]

    dx_err = x[15] - x[3]
    dy_err = x[16] - x[4]
    dz_err = x[17] - x[5]

    lin_vel = [
            dx_err * cz + dy_err * sz,
            -dx_err * sz + dy_err * cz,
            dz_err]

    da_err = x[21] - x[9]
    db_err = x[22] - x[10]
    dg_err = x[23] - x[11]

    ang_vel = [da_err, db_err, dg_err]

    error = np.array([
        pos_err[0], pos_err[1], pos_err[2],
        lin_vel[0], lin_vel[1], lin_vel[2],
        ori_err[0], ori_err[1], ori_err[2],
        ang_vel[0], ang_vel[1], ang_vel[2],
    ])


    error_state = error
    return np.squeeze(error_state)
