import numpy as np
import pytest

from mastermlx import get_backend, set_backend
from mastermlx.robotics import (
    DHLink,
    chain_positions,
    chain_positions_batch,
    geometric_jacobian_batch,
    robotics_backend_report,
)
from mastermlx.robotics.kinematics import _load_cpp_kinematics, forward_kinematics_batch


def _links():
    return [
        DHLink(a=0.35, alpha=0.2, d=0.1, theta=0.05),
        DHLink(a=0.2, alpha=-0.4, d=0.0, theta=0.1, joint_type="prismatic", offset=0.02),
        DHLink(a=0.15, alpha=0.3, d=0.05, theta=-0.2),
    ]


def test_robotics_backend_report_is_consistent():
    report = robotics_backend_report()
    assert report["requested"] == get_backend()
    assert report["active"] in {"numpy", "cython", "cpp"}
    assert isinstance(report["cpp_kinematics"], bool)


def test_cpp_batch_kinematics_matches_numpy_with_base_tool():
    cpp = _load_cpp_kinematics("auto")
    if cpp is None:
        pytest.skip("C++ robotics extension is unavailable")

    base = np.array(
        [[0.0, -1.0, 0.0, 0.2], [1.0, 0.0, 0.0, -0.1], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
    )
    tool = np.array(
        [[1.0, 0.0, 0.0, 0.05], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.02], [0.0, 0.0, 0.0, 1.0]]
    )
    q = np.array([[0.1, 0.03, -0.2], [-0.4, -0.02, 0.25], [0.7, 0.08, 0.1]])
    old = get_backend()
    try:
        set_backend("numpy")
        fk_reference = forward_kinematics_batch(_links(), q, base=base, tool=tool)
        jac_reference = geometric_jacobian_batch(_links(), q, base=base, tool=tool)
        set_backend("auto")
        fk_accelerated = forward_kinematics_batch(_links(), q, base=base, tool=tool)
        jac_accelerated = geometric_jacobian_batch(_links(), q, base=base, tool=tool)
    finally:
        set_backend(old)

    assert np.allclose(fk_accelerated, fk_reference, atol=1e-12)
    assert np.allclose(jac_accelerated, jac_reference, atol=1e-12)


def test_cpp_batch_chain_positions_matches_numpy_with_base_tool():
    cpp = _load_cpp_kinematics("auto")
    if cpp is None or not hasattr(cpp, "chain_positions_batch_dh"):
        pytest.skip("C++ chain-position extension is unavailable")

    base = np.array(
        [[0.0, -1.0, 0.0, 0.2], [1.0, 0.0, 0.0, -0.1], [0.0, 0.0, 1.0, 0.3], [0.0, 0.0, 0.0, 1.0]]
    )
    tool = np.array(
        [[1.0, 0.0, 0.0, 0.05], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.02], [0.0, 0.0, 0.0, 1.0]]
    )
    q = np.array([[0.1, 0.03, -0.2], [-0.4, -0.02, 0.25], [0.7, 0.08, 0.1]])
    old = get_backend()
    try:
        set_backend("numpy")
        reference = np.asarray([chain_positions(_links(), values, base=base, tool=tool) for values in q])
        set_backend("auto")
        accelerated = chain_positions_batch(_links(), q, base=base, tool=tool)
    finally:
        set_backend(old)

    assert np.allclose(accelerated, reference, atol=1e-12)
