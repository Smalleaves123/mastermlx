"""Rotation, quaternion, homogeneous-transform, and point-transform helpers."""

import numpy as np

from mastermlx.robotics import (
    compose_transform,
    euler_to_matrix,
    homogeneous_transform,
    invert_transform,
    matrix_to_euler,
    matrix_to_quaternion,
    quaternion_to_matrix,
    rot_z,
    rpy_to_matrix,
    skew,
    transform_points,
    unskew,
)


rotation = rot_z(np.pi / 4.0)
translation = np.array([1.0, 2.0, 3.0])
transform = homogeneous_transform(rotation, translation)
point = np.array([[1.0, 0.0, 0.0]])
transformed = transform_points(transform, point)
restored = transform_points(invert_transform(transform), transformed)
print("restored_point:", restored[0])
print("identity_check:\n", compose_transform(transform, invert_transform(transform)))

quaternion = matrix_to_quaternion(rotation)
print("quaternion_round_trip:\n", quaternion_to_matrix(quaternion))
print("euler_angles:", matrix_to_euler(rotation))
print("euler_round_trip:\n", euler_to_matrix(*matrix_to_euler(rotation)))
print("rpy_matrix:\n", rpy_to_matrix(0.0, 0.0, np.pi / 4.0))

vector = np.array([1.0, 2.0, 3.0])
print("skew_matrix:\n", skew(vector))
print("unskew_result:", unskew(skew(vector)))

print("transformed_point:", transformed[0])
print("quaternion_wxyz:", quaternion)
