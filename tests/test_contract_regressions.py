import json
import zipfile

import numpy as np
import pytest

from mastermlx.data import cross_val_score
from mastermlx.linear_models import LinearRegression
from mastermlx.robotics import DHLink, RobotModel, RobotResult, URDFRobotModel
from mastermlx.utils import deprecated


def _rewrite_checkpoint_version(source, target, version):
    with zipfile.ZipFile(source) as input_archive, zipfile.ZipFile(
        target, "w", zipfile.ZIP_DEFLATED
    ) as output_archive:
        for info in input_archive.infolist():
            payload = input_archive.read(info.filename)
            if info.filename == "manifest.json":
                manifest = json.loads(payload)
                manifest["library_version"] = version
                payload = json.dumps(manifest, separators=(",", ":")).encode()
            output_archive.writestr(info, payload)


def test_checkpoint_rejects_incompatible_major_and_warns_on_other_versions(tmp_path):
    model = LinearRegression().fit(np.array([[0.0], [1.0]]), np.array([0.0, 1.0]))
    original = tmp_path / "original.mlx"
    minor = tmp_path / "minor.mlx"
    major = tmp_path / "major.mlx"
    model.save(original)
    _rewrite_checkpoint_version(original, minor, "0.2.0")
    _rewrite_checkpoint_version(original, major, "1.0.0")

    with pytest.warns(UserWarning, match="created by mastermlx v0.2.0"):
        restored = LinearRegression.load(minor)
    assert isinstance(restored, LinearRegression)
    with pytest.raises(RuntimeError, match="incompatible"):
        LinearRegression.load(major)


class _BrokenSplitter:
    def split(self, X, y=None, groups=None):
        raise TypeError("internal splitter failure")


def test_cross_validation_does_not_mask_internal_splitter_type_errors():
    X = np.arange(12.0).reshape(6, 2)
    y = np.arange(6.0)

    with pytest.raises(TypeError, match="internal splitter failure"):
        cross_val_score(
            LinearRegression(),
            X,
            y,
            cv=_BrokenSplitter(),
            groups=np.arange(6),
        )


def _dh_robot():
    return RobotModel.from_dh(
        [
            DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
            DHLink(a=1.0, alpha=0.0, d=0.0, theta=0.0),
        ],
        joint_limits=[[-0.5, 0.5], [-0.5, 0.5]],
    )


def test_dh_inverse_kinematics_supports_best_effort_info_and_strict_modes():
    robot = _dh_robot()
    target = np.array([100.0, 100.0, 0.0])

    best_effort = robot.ik(target, joint_values=[4.0, -4.0], max_iter=1)
    info = robot.ik(
        target,
        joint_values=[4.0, -4.0],
        max_iter=1,
        return_info=True,
    )

    assert best_effort.shape == (2,)
    assert isinstance(info, RobotResult)
    assert not info.converged
    assert np.all(info.joint_values >= robot.joint_limits[:, 0])
    assert np.all(info.joint_values <= robot.joint_limits[:, 1])
    with pytest.raises(RuntimeError, match="did not converge"):
        robot.ik(target, max_iter=1, strict=True)


def test_spatial_inverse_kinematics_uses_the_same_failure_contract():
    xml = """
    <robot name="single">
      <link name="base"/><link name="tip"/>
      <joint name="joint" type="revolute">
        <parent link="base"/><child link="tip"/>
        <origin xyz="0 0 0"/><axis xyz="0 0 1"/>
        <limit lower="-0.5" upper="0.5"/>
      </joint>
    </robot>
    """
    robot = URDFRobotModel.from_urdf(xml)
    target = np.array([10.0, 0.0, 0.0])

    best_effort = robot.ik(target, max_iter=1)
    info = robot.ik(target, max_iter=1, return_info=True)

    assert best_effort.shape == (1,)
    assert not info.converged
    with pytest.raises(RuntimeError, match="did not converge"):
        robot.ik(target, max_iter=1, strict=True)


def test_deprecated_decorator_warns_at_the_call_site_and_preserves_metadata():
    @deprecated("new_api", since="0.1.15")
    def old_api(value):
        """Old API documentation."""
        return value + 1

    with pytest.warns(DeprecationWarning, match="use new_api"):
        assert old_api(2) == 3
    assert old_api.__name__ == "old_api"
    assert old_api.__doc__ == "Old API documentation."
