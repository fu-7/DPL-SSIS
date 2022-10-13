import contextlib
import copy
import io
import json
import os
from copy import deepcopy

import numpy as np
import torch
from cvpods.data.build import (SAMPLERS, Infinite, comm, logger,
                               trivial_batch_collator, worker_init_reset_seed)
from cvpods.data.datasets import COCODataset
# from cvpods.data.datasets.paths_route import _PREDEFINED_SPECIAL_COCO
from cvpods.data.detection_utils import (annotations_to_instances,
                                         check_image_size, read_image)


class UnlabeledCOCO:
    def __init__(self, 
                 root='datasets/coco/unlabeled2017', 
                 anno='datasets/coco/annotations/image_info_unlabeled2017.json'):
        with contextlib.redirect_stdout(io.StringIO()):
            self.dataset_dicts = self.load_image_infos(anno,root)
        
    def load_image_infos(self, json_file, root):
        from pycocotools.coco import COCO

        coco_api = COCO(json_file)

        #
