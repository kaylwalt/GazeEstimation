#Uses code from [1], https://gist.github.com/TimSC/8c25ca941d614bf48ebba6b473747d72
#
#
import math
import numpy as np

def trev(angles):
    """input as [theta, phi], [rot around x, rot around y]"""
    angles = [-angles[0], -angles[1]]
    rot_mat_x = np.array([[1, 0, 0],
                          [0, math.cos(angles[0]), -math.sin(angles[0])],
                          [0, math.sin(angles[0]), math.cos(angles[0])]])
    rot_mat_y = np.array([[math.cos(angles[1]), 0, -math.sin(angles[1])],
                          [0, 1, 0],
                          [math.sin(angles[1]), 0, math.cos(angles[1])]])
    start = np.array([[0], [0], [-1]])

    final_rot = np.matmul(rot_mat_x, rot_mat_y)
    answer = np.matmul(final_rot, start)

    return answer

def cam_unit_to_cm(units):
    return (4.0/30.0) * units

def angle_dist(u1, u2):
    return np.arccos(u1.dot(u2))

def t(vec):
    theta = math.asin(-vec[1])
    phi = math.atan2(-vec[0], -vec[2])

    return [theta, phi]

#start of code from [1]
def LinePlaneCollision(planeNormal, planePoint, rayDirection, rayPoint, epsilon=1e-6):
    ndotu = planeNormal.dot(rayDirection)
    if abs(ndotu) < epsilon:
        print("the vector is parralell with the plane")
        return None
    w = rayPoint - planePoint
    si = -planeNormal.dot(w) / ndotu
    Psi = w + si * rayDirection + planePoint
    return Psi
#end of code from [1]
