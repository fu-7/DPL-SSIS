# DPL-SSIS
Semi-supervised Instance Segmentation with Dense Pseudo Labels
# Requirements
[cvpods](https://github.com/Megvii-BaseDetection/cvpods)
# Get Started
+ Install cvpods
```
python3 -m pip install 'git+https://github.com/Megvii-BaseDetection/cvpods.git'
# (add --user if you don't have permission)

# Or, to install it from a local clone:
git clone https://github.com/Megvii-BaseDetection/cvpods.git
python3 -m pip install -e cvpods

```
+ Prepare dataset
```
cd /path/to/cvpods/datasets
ln -s /path/to/your/coco/dataset coco
```
+ Training
```
cd /condinst
pods_train --dir .

```
