# -*- coding: utf-8 -*-

import numpy as np
import theano
import theano.tensor as T
from theano.compat.six.moves import zip as izip


def get_sgd_momentum_update(params, grads, lr, mom, grad_bound):
    """
    SGD + Momentum法
    g(t) = -lr * g(t) + mom * g(t-1)
    W(t+1) = W(t) + g(t)
    """
    prev_grads = []
    for p in params:
        prev_g = theano.shared(np.zeros(p.get_value().shape,
                               dtype=theano.config.floatX), borrow=True)
        prev_grads.append(prev_g)
    assert len(prev_grads) == len(grads)
    updates = []
    for i, (p, g, prev_g) in enumerate(zip(params, grads, prev_grads)):
        updates.append((prev_g,
                        - lr * T.clip(g, -grad_bound, grad_bound)
                        + mom * prev_g))
        updates.append((p, p + prev_g))
    return updates

def get_adadelta_update(params, grads, rho, eps):
    # E[g^2]_{t-1}
    E_g_square = []
    for p in params:
        tmp = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX), borrow=True)
        E_g_square.append(tmp)
    # E[g^2]_t = rho * E[g^2]_{t-1} + (1 - rho) * g_t^2
    E_g_square_next = []
    for e, g in izip(E_g_square, grads):
        tmp = rho * e + (1.0 - rho) * (g**2)
        E_g_square_next.append(tmp)
    # E[dW^2]_{t-1}
    E_dW_square = []
    for p in params:
        tmp = theano.shared(np.zeros(p.get_value().shape, dtype=theano.config.floatX), borrow=True)
        E_dW_square.append(tmp)
    # dW_t = - {sqrt(E[dW^2]_t + eps) / sqrt(E[g^2]_t + eps)} * g_t
    dW = []
    for ew, eg, g in izip(E_dW_square, E_g_square, grads):
        tmp = - (T.sqrt(ew + eps) / T.sqrt(eg + eps)) * g
        dW.append(tmp)
    # E[dW^2]_t = rho * E[dW^2]_{t-1} + (1 - rho) * dW_t^2
    E_dW_square_next = []
    for ew, d in izip(E_dW_square, dW):
        tmp = rho * ew + (1.0 - rho) * (d**2)
        E_dW_square_next.append(tmp)

    E_g_square_updates = zip(E_g_square, E_g_square_next)
    E_dW_square_updates = zip(E_dW_square, E_dW_square_next)
    params_updates = []
    for p, d in izip(params, dW):
        # W_t = W_{t-1} + dW
        params_updates.append((p, p + d))
    return E_g_square_updates + E_dW_square_updates + params_updates
