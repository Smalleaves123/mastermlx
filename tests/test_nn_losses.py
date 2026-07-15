import numpy as np

from mastermlx.neural_net import Adam, AdamW, CrossEntropyLoss, MSELoss, RMSProp, SGD


def test_mse_and_cross_entropy_losses_work():
    mse = MSELoss()
    y_true = np.array([[1.0], [2.0]])
    y_pred = np.array([[0.0], [2.5]])
    assert np.isclose(mse(y_true, y_pred), 0.625)
    assert mse.grad(y_true, y_pred).shape == y_pred.shape

    ce = CrossEntropyLoss(from_logits=True)
    target = np.array([[1.0, 0.0], [0.0, 1.0]])
    logits = np.array([[3.0, 1.0], [0.5, 2.0]])
    loss = ce(target, logits)
    grad = ce.grad(target, logits)
    assert loss > 0.0
    assert grad.shape == logits.shape


def test_optimizers_update_parameters():
    param = np.array([1.0, -1.0])
    grad = np.array([0.1, -0.2])

    sgd = SGD(lr=0.5)
    out = sgd.update(param, grad, "w")
    assert np.allclose(out, np.array([0.95, -0.9]))

    rms = RMSProp(lr=0.1)
    out2 = rms.update(param, grad, "w")
    assert out2.shape == param.shape

    adam = Adam(lr=0.1)
    out3 = adam.update(param, grad, "w")
    assert out3.shape == param.shape


def test_adam_step_counter_advances_once_per_parameter_group():
    for optimizer in (Adam(lr=0.01), AdamW(lr=0.01)):
        optimizer.begin_step()
        optimizer.update(np.ones(2), np.ones(2), "w")
        optimizer.update(np.ones(2), np.ones(2), "b")
        optimizer.end_step()
        assert optimizer._t == 1

        optimizer.begin_step()
        optimizer.update(np.ones(2), np.ones(2), "w")
        optimizer.end_step()
        assert optimizer._t == 2
