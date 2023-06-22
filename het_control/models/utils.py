#  Copyright (c) 2024.
#  ProrokLab (https://www.proroklab.org/)
#  All rights reserved.

import torch

from het_control.utils import clamp_preserve_gradients


def squash(loc, action_spec, clamp: bool):
    """
    Squash loc into the action_spec bounds
    """
    f = clamp_squash if clamp else tanh_squash
    return f(loc, action_spec)


def tanh_squash(loc, action_spec):
    tanh_loc = torch.nn.functional.tanh(loc)
    scale = (action_spec.space.high - action_spec.space.low) / 2
    add = (action_spec.space.high + action_spec.space.low) / 2
    return tanh_loc * scale + add


def clamp_squash(loc, action_spec):
    loc = clamp_preserve_gradients(loc, action_spec.space.low, action_spec.space.high)
    return loc
