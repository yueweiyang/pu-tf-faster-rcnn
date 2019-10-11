import os
import numpy as np
from xml.etree import cElementTree as ET  

def product_sum(dict):
    product_sum = 0
    for key in dict.keys():
        product_sum += int(key)*train_bbox_nimgage_dict[key]
    return product_sum

year = 2012

train_list_path = '../../data/VOCdevkit/VOC{}/ImageSets/Main/trainval.txt'.format(year)
with open(train_list_path) as f:
    train_list = f.read().split()
f.close()

annotation_list_dir = '../../data/VOCdevkit/VOC{}/Annotations/'.format(year)
annotation_list = os.listdir(annotation_list_dir)

n_bboxes = 0
for annotation in annotation_list:
    if annotation[:-4] in train_list:
        file_path = os.path.join(annotation_list_dir,annotation)
        tree = ET.parse(file_path)
        root = tree.getroot()
        nbbox = len(root.findall('object'))
        n_bboxes +=nbbox

annotation_save_dir = '../../data/VOCdevkit/VOC{}/'.format(year)
bbox_ratio = [1.0,0.9,0.8,0.7,0.6,0.5,0.4]

for ratio in bbox_ratio:
    count_del_box = 0
    train_bbox_nimgage_dict = {}
    annotation_list = list(np.random.permutation(annotation_list))
    for annotation in annotation_list:
        file_path = os.path.join(annotation_list_dir,annotation)
        tree = ET.parse(file_path)
        if annotation[:-4] in train_list:
            root = tree.getroot()
            nbbox = len(root.findall('object'))
            if count_del_box < int((1-ratio)*n_bboxes):
                if count_del_box+nbbox-1>=(1-ratio)*n_bboxes:
                    n_del_box = int((1-ratio)*n_bboxes)-count_del_box
                else:
                    n_del_box = nbbox-1
                for delete in range(n_del_box):
                    root.remove(root.find('object'))
                count_del_box +=n_del_box
            nbbox = len(root.findall('object'))
            if nbbox not in train_bbox_nimgage_dict:
                train_bbox_nimgage_dict[nbbox] = 1
            else:
                train_bbox_nimgage_dict[nbbox] += 1
        if not os.path.isdir(os.path.join(annotation_save_dir,'Annotations_{}'.format(ratio))):
            os.mkdir(os.path.join(annotation_save_dir,'Annotations_{}'.format(ratio)))
        tree.write(os.path.join(annotation_save_dir,'Annotations_{}'.format(ratio),annotation))
    print(os.path.join(annotation_save_dir,'Annotations_{}'.format(ratio)),' has {} annotations'.format(product_sum(train_bbox_nimgage_dict)))
