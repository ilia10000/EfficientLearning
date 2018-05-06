# -*- coding: utf-8 -*-
"""
Created on Sat Apr 21 18:37:03 2018

@author: Ilia
"""

from gym.envs.registration import register

register(
    id='classification-v0',
    entry_point='gym_classification.envs:ClassificationEnv',
)
register(
    id='classification-extrahard-v0',
    entry_point='gym_foo.envs:ClassificationExtraHardEnv',
)