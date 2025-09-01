from deepforest import main, get_data
from deepforest.visualize import plot_results
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import utm
import geojson
import os
from pathlib import Path


def image_standardization(image_path: str) -> str:
    image = cv2.imread(image_path)

    hue_standard = 80.0
    sat_standard = 50.0
    light_standard = 80.0

    image_hls = cv2.cvtColor(image, cv2.COLOR_BGR2HLS).astype(np.float32)
    (h, l, s) = cv2.split(image_hls)

    # calc adjustments
    h_mean = h.mean()
    s_mean = s.mean()
    l_mean = l.mean()
    print(f'Source image: h_mean {h_mean}, l_mean {l_mean}, s_mean {s_mean}')
    hue_adj = 1 - ((h_mean - hue_standard) / h_mean)
    sat_adj = 1 - ((s_mean - sat_standard) / s_mean)
    light_adj = 1 - ((l_mean - light_standard) / l_mean)

    # adjust h, s, v
    h = np.clip(h * hue_adj, 0, 255)
    s = np.clip(s * sat_adj, 0, 255)
    l = np.clip(l * light_adj, 0, 255)
    print(f'Adjusted image: h_mean {h.mean()}, l_mean {l.mean()}, s_mean {s.mean()}')

    image_hls = cv2.merge([h, l, s])

    new_image_path = image_path.split('.')[0] + '.jpg'

    cv2.imwrite(new_image_path, cv2.cvtColor(image_hls.astype('uint8'), cv2.COLOR_HLS2BGR))

    return new_image_path

def postprocess_tree_predictions(image_path: str, predictions_gdf: pd.DataFrame) -> pd.DataFrame:
    score_threshold = 0.2
    gray_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    gray_value_threshold = 105
    processed_predictions = []

    for prediction in (pbar := tqdm(predictions_gdf.values.tolist())):
        pbar.set_description('Prostprocessing... ')
        x1, y1, x2, y2 = [int(coord) for coord in prediction[0:4]]
        mean_gray_value = gray_img[y1:y2, x1:x2].mean()
        score = prediction[5]
        if mean_gray_value < gray_value_threshold and score > score_threshold:
            processed_predictions.append(prediction)

    return pd.DataFrame(processed_predictions,
                               columns=['xmin', 'ymin', 'xmax', 'ymax', 'label', 'score', 'image_path', 'geometry'])

def convert_predictions_to_utm_lat_lon_coords(predictions: pd.DataFrame, image_path: str):
    """
    Convert pixel coordinates to UTM coordinates.
    """
    predictions_list = pd.DataFrame(predictions).values.tolist()
    image_name = image_path.split('/')[-1].split('.')[0]
    year = image_name.split('_')[-1]+'-06-01 00:00:00'

    # check reference utm coordinate pair for file
    # 20 cm per pixel -> 5 pixel equals 1m
    # East := x, North := y
    # reference utm is bottom left corner of image (y, 0)

    # GET REFERENCE UTM
    reference_east = image_name.split('_')[2]+'000'
    reference_north = image_name.split('_')[3]+'000'

    coord_list = []
    tree_point_list = []
    # CONVERT XY to UTM
    for prediction in predictions_list:
        x1, y1, x2, y2 = prediction[:4]
        x_center = (x1 + x2) // 2
        y_center = (y1 + y2) // 2
        rel_meters_bl_x = x_center // 5
        rel_meters_bl_y = (10000 - y_center) // 5
        utm_east = int(reference_east) + rel_meters_bl_x
        utm_north = int(reference_north) + rel_meters_bl_y

        lat, lon = utm.to_latlon(utm_east, utm_north, 32, 'U')
        coord_list.append([utm_east, utm_north, lat, lon, year])
        tree_point_list.append(geojson.Point((lat, lon)))

    pd.DataFrame(coord_list, columns=['utm_east', 'utm_north', 'latitude', 'longitude', 'date']).to_csv(
        f'/home/debler/Projekte/harz_analysis/predicted_tree_locations/{image_name}.csv', index=False
    )
    with open(f'/home/debler/Projekte/harz_analysis/predicted_tree_locations/{image_name}.geojson', 'w') as geojson_file:
        geojson.dump(tree_point_list, geojson_file, indent=4)

def check_if_image_jpg(image_path: str) -> str:
    if image_path.endswith('.jpg'):
        return image_path
    elif image_path.split('/')[-1].replace('.tif','.jpg') in os.listdir(Path(image_path).parent):
        return image_path.replace('.tif','.jpg')
    else:
        image = cv2.imread(image_path)
        new_image_path = image_path.split('.')[0]+'.jpg'
        cv2.imwrite(new_image_path, image)
        return new_image_path

def join_annotation_csvs():
    root_dir = '/home/debler/Projekte/harz_analysis/predicted_tree_locations'
    csv_paths = []
    for csv_name in os.listdir(root_dir):
        if csv_name.split('.')[-1] == 'csv' and '2014' in csv_name:
            csv_paths.append(os.path.join(root_dir, csv_name))

    tree_location_data_list = []
    for csv_path in csv_paths:
        with open(csv_path,'r') as file:
            tree_location_data_list.append(file.read().replace('utm_east,utm_north,latitude,longitude,date\n',''))

    with open('/home/debler/Projekte/harz_analysis/predicted_tree_locations.csv', 'a') as file:
        for tree_location_data in tree_location_data_list:
            file.write(tree_location_data)

def predict_tree_locations(path: str):
    #model = main.deepforest()
    #model.load_model(model_name='weecology/deepforest-tree', revision='main')
    model = main.deepforest.load_from_checkpoint('weights/weights_100_epochs_w_threshold.pl')

    path = check_if_image_jpg(path)
    image_path = get_data(path)

    predictions_gdf = model.predict_tile(image_path, patch_size=250, patch_overlap=0.25)

    # postprocessing
    predictions = postprocess_tree_predictions(image_path, predictions_gdf)
    print(f'{len(predictions)} predictions after postprocessing')

    plot_results(results=predictions, savedir='predicted_tree_locations', image=np.array(Image.open(image_path)))

    convert_predictions_to_utm_lat_lon_coords(predictions, path)

# root_dir = '/home/debler/Projekte/s-a_ortho'
# tile_paths = []
# for tile_name in os.listdir(root_dir):
#     if tile_name.split('.')[-1] == 'tif' and '2014' in tile_name:
#         tile_paths.append(os.path.join(root_dir, tile_name))
#
# for path in tile_paths:
#     predict_tree_locations(path)


join_annotation_csvs()