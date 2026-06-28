import mastermlx

from mastermlx.math_tools import (
    ARModel,
    entropy,
    precision_score,
    multi_head_attention,
    pairwise_distance,
    pairwise_kernel,
    scaled_dot_product_attention,
    sinusoidal_positional_encoding,
)


def test_math_tools_package_is_unified_and_public():
    assert hasattr(mastermlx, "math_tools")
    assert ARModel is not None
    assert entropy is not None
    assert precision_score is not None
    assert pairwise_distance is not None
    assert pairwise_kernel is not None
    assert scaled_dot_product_attention is not None
    assert multi_head_attention is not None
    assert sinusoidal_positional_encoding is not None
