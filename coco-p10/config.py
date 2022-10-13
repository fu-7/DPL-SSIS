from cvpods.configs.condinst_config import CondInstConfig

from augmentations import WeakAug,StrongAug
from dataset import PartialCOCO, FullCOCO
    
_config_dict = dict(
    DATASETS=dict(
        SUPERVISED=[
            (PartialCOCO,dict(
                percentage=10,
                seed=1,
                supervised=True,
                sup_file='../COCO_Division/COCO_supervision.txt'
            )),
        ],
        UNSUPERVISED=[
            (PartialCOCO,dict(
                percentage=10,
                seed=1,
                supervised=False,
                sup_file='../COCO_Division/COCO_supervision.txt'
            )),
        ],
        TEST=
        ("coco_2017_val",),
    ),
    MODEL=dict( 
        # WEIGHTS='detectron2://ImageNetPretrained/MSRA/R-50.pkl',
        WEIGHTS='/root/fuqi/DenseTeacher/weights/R-50.pkl',
        RESNETS=dict(DEPTH=50),
        FCOS=dict(
            QUALITY_BRANCH='iou',
            CENTERNESS_ON_REG=Tru
