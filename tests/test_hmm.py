import numpy as np

from mastermlx import get_backend, set_backend
from mastermlx.probabilistic import HMM


def test_hmm_fits_short_sequences():
    seqs = [
        np.array([0, 0, 1, 1, 0]),
        np.array([0, 1, 1, 1, 0]),
        np.array([1, 1, 0, 0, 0]),
    ]

    model = HMM(n_states=2, n_obs=2, random_state=0)
    model.fit(seqs, n_iter=20)

    assert model.start_.shape == (2,)
    assert model.trans_.shape == (2, 2)
    assert model.emit_.shape == (2, 2)
    assert len(model.loglik_) >= 1


def test_hmm_viterbi_returns_path():
    seqs = [
        np.array([0, 0, 1, 1, 0]),
        np.array([0, 1, 1, 1, 0]),
        np.array([1, 1, 0, 0, 0]),
    ]

    model = HMM(n_states=2, n_obs=2, random_state=0)
    model.fit(seqs, n_iter=10)
    path = model.predict(np.array([0, 1, 1, 0]))

    assert path.shape == (4,)
    assert set(path.tolist()) <= {0, 1}


def test_hmm_emissions_accumulate_duplicate_observations():
    model = HMM(n_states=2, n_obs=3, random_state=0)
    model.start_ = np.array([0.5, 0.5])
    model.trans_ = np.array([[0.5, 0.5], [0.5, 0.5]])
    model.emit_ = np.full((2, 3), 1.0 / 3.0)

    seq = np.array([0, 2, 0])
    model._check_seq = lambda x: np.asarray(x, dtype=int)
    model._forward = lambda x: np.zeros((x.size, model.n_states))
    model._backward = lambda x: np.zeros((x.size, model.n_states))

    model.fit([seq], n_iter=1)

    assert np.allclose(model.emit_[:, 0], 2.0 / 3.0)
    assert np.allclose(model.emit_[:, 2], 1.0 / 3.0)


def test_hmm_cpp_and_numpy_dynamic_programs_match():
    model = HMM(n_states=2, n_obs=2, random_state=4)
    model._init_params()
    seq = np.array([0, 1, 1, 0, 1])
    old = get_backend()
    try:
        set_backend("auto")
        forward_cpp = model._forward(seq)
        backward_cpp = model._backward(seq)
        path_cpp = model.predict(seq)
        set_backend("numpy")
        forward_numpy = model._forward(seq)
        backward_numpy = model._backward(seq)
        path_numpy = model.predict(seq)
    finally:
        set_backend(old)

    assert np.allclose(forward_cpp, forward_numpy)
    assert np.allclose(backward_cpp, backward_numpy)
    assert np.array_equal(path_cpp, path_numpy)
