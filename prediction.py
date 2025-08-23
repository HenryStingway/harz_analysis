from deepforest import main, get_data
from deepforest.visualize import plot_results
import cv2
from tqdm import tqdm
import pandas as pd
import numpy as np
from PIL import Image
import utm
import geojson


def postprocess_tree_predictions(image_path: str, predictions_gdf: pd.DataFrame) -> pd.DataFrame:
    score_threshold = 0.2
    gray_img = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2GRAY)
    gray_value_threshold = np.mean(gray_img)
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

    pd.DataFrame(coord_list, columns=['utm_east', 'utm_north', 'latitude', 'longitude', 'date']).to_csv(f'{image_name}.csv', index=False)
    with open(f'{image_name}.geojson', 'w') as geojson_file:
        geojson.dump(tree_point_list, geojson_file, indent=4)

def check_if_image_jpg(image_path: str) -> str:
    if image_path.endswith('.jpg'):
        return image_path
    else:
        image = cv2.imread(image_path)
        new_image_path = image_path.split('.')[0]+'.jpg'
        cv2.imwrite(new_image_path, image)
        return new_image_path

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


path_list = [
    's-a_ortho/dop20rgbi_32_612_5736_2_st_2014.tif',
    's-a_ortho/dop20rgbi_32_612_5736_2_st_2016.tif'
]
for path in path_list:
    predict_tree_locations(path)


# join predictions csv in predicted_tee_locations.csv
with open('dop20rgbi_32_612_5736_2_st_2014.csv','r') as file:
    tree_locations_2014 = file.read()
with open('dop20rgbi_32_612_5736_2_st_2016.csv','r') as file:
    tree_locations_2016 = file.read()
with open('predicted_tree_locations.csv','a') as file:
    file.write(tree_locations_2014)
    file.write(tree_locations_2016)