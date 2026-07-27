# mastermlx examples

This directory is the public, copy-and-run tutorial for `mastermlx`. The
examples use the normal installed Python package and do not depend on a local
Conda environment, an absolute filesystem path, or the repository internals.

Examples `00` through `09` target the public API released in
`mastermlx==0.1.13`. Example `10` demonstrates the newer signal-health APIs in
the current source tree; run it after installing this checkout with
`python -m pip install -e .`, or wait for the next PyPI release before using
that example with an installed package.

## 1. Install

Python 3.10 or newer is recommended.

```bash
python -m pip install "mastermlx==0.1.13"
```

To follow the newest PyPI release instead:

```bash
python -m pip install --upgrade mastermlx
```

## 2. Run the tutorial

From this directory:

```bash
python 00_quickstart.py
```

Each example is independently executable. The files are tutorials, not part of
the package's automated test suite.

## 3. Five-minute quickstart

```python
import numpy as np
from mastermlx.robotics import DHLink, RobotModel

robot = RobotModel.from_dh([
    DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
    DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
])

q = np.array([0.2, -0.1])
pose = robot.fk(q)                 # (4, 4) homogeneous transform
jacobian = robot.jacobian(q)      # (6, n_joints) geometric Jacobian
position = pose[:3, 3]             # (3,) end-effector position

print(position)
```

The first three rows of the Jacobian are linear velocity and the last three
rows are angular velocity. Joint angles are in radians and distances use the
same unit as the DH link lengths.

## 4. Public API guide

### Robot model construction

```python
robot = RobotModel.from_dh(
    links,
    name="arm",
    base=None,     # optional 4 x 4 base transform
    tool=None,     # optional 4 x 4 tool transform
)
```

`DHLink` uses the standard DH fields:

| Field | Meaning |
| --- | --- |
| `a` | link length |
| `alpha` | link twist, radians |
| `d` | link offset |
| `theta` | fixed joint angle offset |
| `joint_type` | `"revolute"` or `"prismatic"` |
| `offset` | runtime joint offset |

### Kinematics and Jacobians

| API | Input | Output |
| --- | --- | --- |
| `robot.fk(q)` | `(n_joints,)` | `(4, 4)` transform |
| `robot.fk(q, return_all=True)` | `(n_joints,)` | final transform and frame list |
| `robot.fk_batch(Q)` | `(n_samples, n_joints)` | `(n_samples, 4, 4)` |
| `robot.positions(q)` | `(n_joints,)` | chain-frame positions |
| `robot.jacobian(q)` | `(n_joints,)` | `(6, n_joints)` |
| `robot.jacobian_batch(Q)` | `(n_samples, n_joints)` | `(n_samples, 6, n_joints)` |
| `robot.ik(target, q0)` | `(3,)` or `(4,4)` target | `(n_joints,)` solution |

The functional equivalents are `forward_kinematics`,
`geometric_jacobian`, and `inverse_kinematics`.

### Coordinate transforms

```python
T = homogeneous_transform(rotation_3x3, translation_3)
T_inverse = invert_transform(T)
world_points = transform_points(T, local_points)
q_wxyz = matrix_to_quaternion(rotation_3x3)
rotation_3x3 = quaternion_to_matrix(q_wxyz)
```

Use `rot_x`, `rot_y`, `rot_z`, `rpy_to_matrix`, `euler_to_matrix`, and
`compose_transform` to build transforms. `skew(v)` and `unskew(M)` convert
between a 3-vector and its skew-symmetric matrix.

### Trajectories

```python
times, positions, velocities, accelerations = sample_joint_trajectory(
    q0, qf, duration=2.0, num_samples=100, kind="quintic"
)
```

For multiple waypoints use `sample_joint_trajectory_segments`. For a complete
path plus time parameterization use `plan_joint_trajectory`. The returned
arrays have shapes `(n_samples,)` and `(n_samples, n_joints)`.

### URDF and workcell workflows

```python
robot = RobotModel.from_urdf(xml_text)
workcell = RobotWorkcell(robot)
result = workcell.plan_cartesian_task(
    targets,
    q_start,
    steps_per_segment=10,
    check_collisions=False,
)
joint_path = result["joint_path"]
```

`RobotExperiment` is a lightweight workflow wrapper around a model. Workcell
results are mapping-compatible dictionaries; inspect their keys before using
the arrays in an application.

## 5. Example index

| File | What it teaches |
| --- | --- |
| `00_quickstart.py` | Minimal model construction and FK |
| `01_kinematics.py` | DH, FK, Jacobian, batch APIs, and IK |
| `02_transforms.py` | Rotation, quaternion, and homogeneous transforms |
| `03_trajectory.py` | Time scaling, paths, smoothing, and sampling |
| `04_urdf_experiment.py` | URDF conversion and experiment comparison |
| `05_workcell.py` | TCP solving and Cartesian task planning |
| `06_basic_ml.py` | The same fit/transform/predict style outside robotics |
| `07_state_estimation.py` | Planar pose filtering and sensor updates |
| `08_public_api_catalog.py` | Public robotics API names and recommended entry points |
| `09_signal_condition_monitoring.py` | Vibration quality, features, and streaming detection |
| `10_signal_health_monitor.py` | Windowed features, health scores, and threshold alerts |

The catalog is intentionally lightweight: it prints the callable signatures;
the preceding files show the corresponding calls with real data.

## 6. Common mistakes

- Pass joint configurations with shape `(n_joints,)`; pass batches with shape
  `(n_samples, n_joints)`.
- Pass a 3-vector for position-only IK or a 4x4 transform for pose IK.
- Keep angle units in radians.
- Keep the frame convention consistent: transforms map local coordinates into
  their parent/world frame.
- Install the package before running examples. When working from a cloned
  repository, run the scripts from this directory so the installed package is
  resolved predictably.

The local development branch may contain newer APIs such as constrained IK,
`ik_batch`, and differential IK. Those are intentionally not used here until
they are included in a public PyPI release.
