from functools import partial

import numpy as np

from ...utils import common_utils
from . import augmentor_utils, database_sampler


class DataAugmentor(object):
    def __init__(self, root_path, augmentor_configs, class_names, logger=None):
        self.root_path = root_path
        self.class_names = class_names
        self.logger = logger

        self.data_augmentor_queue = []
        aug_config_list = augmentor_configs if isinstance(augmentor_configs, list) \
            else augmentor_configs.AUG_CONFIG_LIST

        for cur_cfg in aug_config_list:
            if not isinstance(augmentor_configs, list):
                if cur_cfg.NAME in augmentor_configs.DISABLE_AUG_LIST:
                    continue
            cur_augmentor = getattr(self, cur_cfg.NAME)(config=cur_cfg)
            self.data_augmentor_queue.append(cur_augmentor)

    def gt_sampling(self, config=None):
        db_sampler = database_sampler.DataBaseSampler(
            root_path=self.root_path,
            sampler_cfg=config,
            class_names=self.class_names,
            logger=self.logger
        )
        return db_sampler

    def __getstate__(self):
        d = dict(self.__dict__)
        del d['logger']
        return d

    def __setstate__(self, d):
        self.__dict__.update(d)
   
    def random_world_flip(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_flip, config=config)
        gt_seg, points = data_dict['labels_seg'], data_dict['points']
        observations = data_dict.get('observations', None)
        for cur_axis in config['ALONG_AXIS_LIST']:
            assert cur_axis in ['x', 'y']
            gt_seg, points, observations = getattr(augmentor_utils, 'random_flip_along_%s' % cur_axis)(
                gt_seg, points, observations
            )

        data_dict['labels_seg'] = gt_seg
        data_dict['points'] = points
        if observations is not None:
            data_dict['observations'] = observations
        return data_dict

    def random_world_rotation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_rotation, config=config)
        rot_range = config['WORLD_ROT_ANGLE']
        observations = data_dict.get('observations', None)
        if not isinstance(rot_range, list):
            rot_range = [-rot_range, rot_range]
        gt_seg, points, observations = augmentor_utils.global_rotation(
            data_dict['labels_seg'], data_dict['points'], observations, rot_range=rot_range
        )

        data_dict['labels_seg'] = gt_seg
        data_dict['points'] = points

        if observations is not None:
            data_dict['observations'] = observations

        return data_dict

    def random_world_scaling(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_scaling, config=config)
        observations = data_dict.get('observations', None)
        gt_seg, points, observations = augmentor_utils.global_scaling(
            data_dict['labels_seg'], data_dict['points'], observations, config['WORLD_SCALE_RANGE']
        )
        data_dict['labels_seg'] = gt_seg
        data_dict['points'] = points

        if observations is not None:
            data_dict['observations'] = observations

        return data_dict

    def random_world_translation(self, data_dict=None, config=None):
        if data_dict is None:
            return partial(self.random_world_translation, config=config)
        observations = data_dict.get('observations', None)
        gt_seg, points, observations = augmentor_utils.global_translate(
            data_dict['labels_seg'], data_dict['points'], observations, config['WORLD_TRANSLATE_RANGE']
        )
        data_dict['labels_seg'] = gt_seg
        data_dict['points'] = points

        if observations is not None:
            data_dict['observations'] = observations

        return data_dict
