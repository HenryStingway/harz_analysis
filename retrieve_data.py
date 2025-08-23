
import os
import json
from pathlib import Path
import pandas as pd
import cv2
import numpy as np

# dtools.download(dataset='NeonTreeEvaluation: RGB', dst_dir='~/dataset-ninja/')

def retrieve_training_data():
    neon_tree_ds_path = '/home/debler/dataset-ninja/neontreeevaluation:-rgb'

    annotations = []

    for folder in ['evaluation']:
        anno_files = os.listdir(f'{neon_tree_ds_path}/{folder}/ann')
        train_size = int(len(anno_files) * 0.8)
        val_size = int(len(anno_files) * 0.2)
        # training
        for annotation_file in anno_files[:train_size]:
            f = open(f'{neon_tree_ds_path}/{folder}/ann/{annotation_file}')
            ann_data = json.load(f)
            f.close()
            image_path = Path(annotation_file).stem
            img = cv2.imread(f'{neon_tree_ds_path}/{folder}/img/{image_path}')
            img_name = image_path.split('.')[0]
            cv2.imwrite(f'neon_tree_dataset/train/img/{img_name}.jpg', img)
            for object_data in ann_data['objects']:
                x1,y1 = object_data['points']['exterior'][0]
                x2,y2 = object_data['points']['exterior'][1]
                annotations.append([f'{img_name}.jpg',x1,y1,x2,y2, object_data['classTitle']])
        pd.DataFrame(annotations, columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']).to_csv(
            f'neon_tree_dataset/train/annotations.csv', index=False)

        annotations = []
        # validation
        for annotation_file in anno_files[-val_size:]:
            f = open(f'{neon_tree_ds_path}/{folder}/ann/{annotation_file}')
            ann_data = json.load(f)
            f.close()
            image_path = Path(annotation_file).stem
            img = cv2.imread(f'{neon_tree_ds_path}/{folder}/img/{image_path}')
            img_name = image_path.split('.')[0]
            cv2.imwrite(f'neon_tree_dataset/val/img/{img_name}.jpg', img)
            for object_data in ann_data['objects']:
                x1,y1 = object_data['points']['exterior'][0]
                x2,y2 = object_data['points']['exterior'][1]
                annotations.append([f'{img_name}.jpg',x1,y1,x2,y2, object_data['classTitle']])
        pd.DataFrame(annotations, columns=['image_path', 'xmin', 'ymin', 'xmax', 'ymax', 'label']).to_csv(
            f'neon_tree_dataset/val/annotations.csv', index=False)


def visualize_train_example():
    img_path = 'neon_tree_dataset/train/img/BONA_006_2019.jpg'
    annotations = pd.read_csv('neon_tree_dataset/train/annotations.csv')
    img_name = img_path.split('/')[-1]
    img_annos = annotations.query('image_path == @img_name').values.tolist()
    bbox_img = cv2.imread(img_path).copy()
    for anno in img_annos:
        x1 = anno[1]
        y1 = anno[2]
        x2 = anno[3]
        y2 = anno[4]
        cv2.rectangle(bbox_img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    cv2.imwrite('bbox_img.jpg', bbox_img)

# img = cv2.imread('s-a_ortho/dop20rgbi_32_612_5736_2_st_2024.tif')
# cv2.imwrite('s-a_ortho/dop20rgbi_32_612_5736_2_st_2024.jpg', img)


# with open('dop20rgbi_32_612_5736_2_st_2018.csv','r') as file:
#     tree_locations_2018 = file.read()
# with open('dop20rgbi_32_612_5736_2_st_2024.csv','r') as file:
#     tree_locations_2024 = file.read()
# with open('predicted_tree_locations.csv','a') as file:
#     file.write(tree_locations_2018)
#     file.write(tree_locations_2024)