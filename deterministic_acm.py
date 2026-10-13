"""2-D morphological Chan-Vese with a local, explicit curvature-smoothing phase.

The region update and four line structuring elements follow scikit-image's
morphsnakes implementation. Unlike its module-global curvature cycle, phase
here depends only on iteration, so processing order cannot change a detection.
"""
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion

HORIZONTAL = np.zeros((3, 3), dtype=bool)
HORIZONTAL[1, :] = True
LINES = (np.eye(3, dtype=bool), HORIZONTAL, np.fliplr(np.eye(3, dtype=bool)), HORIZONTAL.T)


def sup_inf(u):
    return np.stack([binary_erosion(u, line) for line in LINES]).max(axis=0).astype(np.int8)


def inf_sup(u):
    return np.stack([binary_dilation(u, line) for line in LINES]).min(axis=0).astype(np.int8)


def chan_vese(image, num_iter, init_level_set, lambda1=1.0, lambda2=1.5, start_iteration=0):
    image = np.asarray(image)
    if image.ndim != 2 or min(image.shape) < 2:
        raise ValueError("ACM expects a 2-D crop at least 2x2")
    u = np.int8(np.asarray(init_level_set) > 0)
    if u.shape != image.shape:
        raise ValueError("Initial mask and image shapes differ")
    for iteration in range(start_iteration, start_iteration + num_iter):
        c0 = (image*(1-u)).sum() / float((1-u).sum()+1e-8)
        c1 = (image*u).sum() / float(u.sum()+1e-8)
        magnitude = np.abs(np.gradient(u)).sum(axis=0)
        update = magnitude*(lambda1*(image-c1)**2 - lambda2*(image-c0)**2)
        u[update < 0] = 1
        u[update > 0] = 0
        u = sup_inf(inf_sup(u)) if iteration % 2 == 0 else inf_sup(sup_inf(u))
    return u
