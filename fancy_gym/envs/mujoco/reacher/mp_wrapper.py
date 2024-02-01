from typing import Union, Tuple

import numpy as np

from fancy_gym.black_box.raw_interface_wrapper import RawInterfaceWrapper


class MPWrapper(RawInterfaceWrapper):

    mp_config = {
        'ProMP': {},
        'DMP': {
            'phase_generator_kwargs': {
                'alpha_phase': 2,
            },
        },
        'ProDMP': {
            "trajectory_generator_kwargs": {
                'trajectory_generator_type': 'prodmp',
                'duration': 2.0,
                'weights_scale': 0.3,
                'goal_scale': 0.3,
                'auto_scale_basis': True,
                'disable_goal': False,
                'relative_gaol': True,
            },
            "phase_generator_kwargs": {
                'phase_generator_type': 'exp',
                'alpha_phase': 3,
                'tau': 1.5,
            },
            "controller_kwargs": {
                'controller_type': 'motor',
                "p_gains": 1.0,
                "d_gains": 0.1,
            },
            "basis_generator_kwargs": {
                'basis_generator_type': 'prodmp',
                'alpha': 10,
                'num_basis': 5,
                'basis_bandwidth_factor': 3,
            },
            "black_box_kwargs": {
            }
        
        }


    }

    @property
    def context_mask(self):
        return np.concatenate([[False] * self.n_links,  # cos
                               [False] * self.n_links,  # sin
                               [True] * 2,  # goal position
                               [False] * self.n_links,  # angular velocity
                               [False] * 3,  # goal distance
                               # [False],  # step
                               ])

    @property
    def current_pos(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qpos.flat[:self.n_links]

    @property
    def current_vel(self) -> Union[float, int, np.ndarray, Tuple]:
        return self.data.qvel.flat[:self.n_links]
