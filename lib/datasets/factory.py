# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Factory method for easily getting imdbs by name."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

__sets = {}
from datasets.pascal_voc import pascal_voc
from datasets.coco import coco
# from datasets.chest_xrays import chest_xrays
# from datasets.open_images import open_images

import numpy as np

# Set up voc_<year>_<split> 
for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    for ratio_data in [0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
      for ratio_bbox in [0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        name = 'voc_{}_{}_{}_{}'.format(year, split,ratio_data,ratio_bbox)
        __sets[name] = (lambda split=split, year=year, ratio_data=ratio_data, ratio_bbox=ratio_bbox: pascal_voc(split, year, ratio_data, ratio_bbox))

for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    for ratio_data in [0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
      for ratio_bbox in [0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        name = 'voc_{}_{}_{}_{}_diff'.format(year, split,ratio_data,ratio_bbox)
        __sets[name] = (lambda split=split, year=year, ratio_data=ratio_data, ratio_bbox=ratio_bbox: pascal_voc(split, year, ratio_data, ratio_bbox, use_diff= True))

for year in ['2007', '2012']:
  for split in ['train', 'val', 'trainval', 'test']:
    name = 'voc_{}_{}'.format(year, split)
    __sets[name] = (lambda split=split, year=year, ratio_data=1.0, ratio_bbox=1.0: pascal_voc(split, year, ratio_data, ratio_bbox))

# Set up coco_2014_<split>
for year in ['2014']:
  for split in ['train', 'val', 'minival', 'valminusminival', 'trainval']:
    for ratio_data in [0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
      for ratio_bbox in [0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
        name = 'coco_{}_{}_{}_{}'.format(year, split,ratio_data,ratio_bbox)
        __sets[name] = (lambda split=split, year=year, ratio_data=ratio_data, ratio_bbox=ratio_bbox: coco(split, year,ratio_data,ratio_bbox))

for year in ['2014']:
  for split in ['minival']:
    name = 'coco_{}_{}'.format(year,split)
    __sets[name] = (lambda split=split, year=year, ratio_data=1.0, ratio_bbox=1.0:coco(split,year,ratio_data,ratio_bbox))	

# Set up coco_2015_<split>
for year in ['2015']:
  for split in ['test', 'test-dev']:
    for ratio_data in [1.0]:
      for ratio_bbox in [1.0]:
        name = 'coco_{}_{}_{}_{}'.format(year, split,ratio_data,ratio_bbox)
        __sets[name] = (lambda split=split, year=year, ratio_data=ratio_data, ratio_bbox=ratio_bbox: coco(split, year,ratio_data,ratio_bbox))

# for split in ['train', 'val', 'test', 'trainval']:
#     for ratio_data in [1.0]:
#         for ratio_bbox in [1.0]:
#             name = 'chest_xrays_{}_{}_{}'.format(split,ratio_data,ratio_bbox)
#             __sets[name] = (lambda split=split,ratio_data=ratio_data,
#                             ratio_bbox=ratio_bbox:chest_xrays(split,ratio_data,ratio_bbox))
# for split in ['test']:
#     name = 'chest_xrays_{}'.format(split)
#     __sets[name] = (lambda split=split,ratio_data=1.0,ratio_bbox=1.0:chest_xrays(split,ratio_data,ratio_bbox))

    
# for split in ['minitrainval', 'minival', 'trainval','test']:
#     for ratio_data in [1.0]:
#         for ratio_bbox in [1.0]:
#             name = 'open_images_{}_{}_{}'.format(split,ratio_data,ratio_bbox)
#             __sets[name] = (lambda split=split,ratio_data=ratio_data,
#                             ratio_bbox=ratio_bbox:open_images(split,ratio_data,ratio_bbox))
            
# for split in ['minival','test','minitest']:
#     name = 'open_images_{}'.format(split)
#     __sets[name] = (lambda split=split,ratio_data=1.0,ratio_bbox=1.0:open_images(split,ratio_data,ratio_bbox))

    
def get_imdb(name):
  """Get an imdb (image database) by name."""
  if name not in __sets:
    raise KeyError('Unknown dataset: {}'.format(name))
  return __sets[name]()


def list_imdbs():
  """List all registered imdbs."""
  return list(__sets.keys())
