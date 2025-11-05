import configparser
import os
from pathlib import Path
import pandas as pd
import ast


def getArgs():
    parser = configparser.ConfigParser()
    parser.optionxform = lambda option: option  # to preserve case for letters
    parser.read("./setup.cfg")

    arguments = {}
    # Day Parameters
    arguments["frames_skipped"] = int(parser['GENERAL']['frames_skipped'])
    arguments["detection_threshold"] = float(parser['PARAMS_DAY']['detection_threshold'])
    arguments["accuracy_thresh"] = float(parser['PARAMS_DAY']['accuracy_threshold'])
    arguments["detection_threshold_small_assets"] = float(
        parser['PARAMS_DAY']['detection_threshold_small_assets'])  # detection threshold for small assets
    arguments["detection_threshold_linear_assets"] = float(parser['PARAMS_DAY']['detection_threshold_linear_assets'])

    arguments["weightPath"] = parser['PARAMS_DAY']['weightPath']
    arguments["usecases"] = parser['PARAMS_DAY']['usecases']
    arguments["night_usecases"] = parser['PARAMS_NIGHT']['usecases']
    arguments["ignore_assets"] = parser['PARAMS_DAY']['ignore_assets']
    arguments["close_assets"] = parser['PARAMS_DAY']['close_assets']
    arguments["night_configPath"] = parser['PARAMS_NIGHT']['configPath']
    arguments["night_weightPath"] = parser['PARAMS_NIGHT']['weightPath']
    arguments["night_dataPath"] = parser['PARAMS_NIGHT']['metaPath']
    arguments["thresh_Street_Light_NW"] = parser['PARAMS_NIGHT']['thresh_Street_Light_NW']

    # Lane Parameters
    arguments["lane_configPath"] = parser['PARAMS_LANE']['configPath']
    arguments["lane_weightPath"] = parser['PARAMS_LANE']['weightPath']
    arguments["lane_dataPath"] = parser['PARAMS_LANE']['metaPath']
    arguments["lane_detection_threshold"] = parser['PARAMS_LANE']['detection_threshold']
    arguments["lane_accuracy_threshold"] = parser['PARAMS_LANE']['accuracy_threshold']

    arguments["linear_config"] = parser["PARAMS_LINEAR"]["config"]
    arguments["linear_weights"] = parser["PARAMS_LINEAR"]["weights"]
    arguments["linear_detection_threshold"] = float(parser["PARAMS_LINEAR"]["detection_threshold"])
    arguments["linear_accuracy_threshold"] = float(parser["PARAMS_LINEAR"]["detection_threshold"])
    arguments["calculate_linear_asset_area"] = parser['PARAMS_LINEAR'].getboolean('calculate_linear_asset_area')
    arguments["plant_anomaly"] = parser['PARAMS_LINEAR'].getboolean('plant_anomaly')
    arguments["crack_classifier_enable"] = parser['PARAMS_LINEAR'].getboolean('crack_classifier_enable')
    arguments["crack_classifier_weight"] = parser["PARAMS_LINEAR"]["crack_classifier_weight"]
    arguments["ignore_rectification"] = parser["PARAMS_LINEAR"]["ignore_rectification"]

    

    arguments["debug"] = parser['PARAMS_DAY'].getboolean('debug')
    arguments["median_detections"] = parser['GENERAL'].getboolean('median_detections')
    arguments["show_count"] = parser['PARAMS_DAY'].getboolean('show_count')
    arguments["show_video"] = parser['PARAMS_DAY'].getboolean('show_video')
    arguments["write_video"] = parser['PARAMS_DAY'].getboolean('write_video')
    arguments["save_csv"] = parser['PARAMS_DAY'].getboolean('save_csv')
    arguments["save_sql"] = parser['PARAMS_DAY'].getboolean('save_sql')
    arguments["send_mail"] = parser['PARAMS_DAY'].getboolean('send_mail')
    arguments["save_count_video"] = parser['PARAMS_DAY'].getboolean('save_count_video')
    arguments["remove_processed_video"] = parser["GENERAL"].getboolean("remove_processed_video")
    arguments["is_local_gui"] = parser["GENERAL"].getboolean("is_local_gui")

    # General Parameters
    arguments["site_name"] = parser['GENERAL']['site_name']
    arguments["server_ip"] = parser['GENERAL']['server_ip']
    arguments["ocr_model"] = parser['GENERAL']['ocr_model']
    arguments["auto_upload_path"] = parser['GENERAL']['auto_upload_path']
    arguments["save_image_path"] = parser['GENERAL']['save_image_path']
    arguments["python_path"] = parser["GENERAL"]["python_path"]
    arguments["max_speed"] = parser["GENERAL"]["max_speed"]

    arguments['gps_path'] = parser['GENERAL']['gps_path']
    arguments["csv_path"] = parser['GENERAL']['csv_path']
    # create_empty_directory(arguments["csv_path"])
    arguments["video_output_path"] = parser['GENERAL']['video_output_path']
    # create_empty_directory(arguments["video_output_path"])
    arguments["error_video_path"] = parser['GENERAL']["error_video_path"]
    # create_empty_directory(arguments["error_video_path"])

    # DB Parameters
    arguments["user"] = parser['LOCAL_DATABASE']['user']
    arguments["password"] = parser['LOCAL_DATABASE']['password']
    arguments["host"] = parser['LOCAL_DATABASE']['host']
    arguments["database"] = parser['LOCAL_DATABASE']['database']

    # arguments["ALPHA"] = parser['ALPHA']
    arguments["HYPER_PARAMETERS"] = parser['DAY_HYPER_PARAMETERS']
    arguments["NIGHT_HYPER_PARAMETERS"] = parser["NIGHT_HYPER_PARAMETERS"]
    arguments["LINEAR_HYPER_PARAMETERS"] = parser['LINEAR_HYPER_PARAMETERS']
    templist=[]
    for key,value in parser['IGNORE_CHAINAGES'].items():
        # print(key)
        # print(value)
        templist.append(value.split(","))
    # print(templist)
    arguments["IGNORE_CHAINAGES"] = templist
    # create_empty_directory("./logs")
    return arguments


def create_empty_directory(path):
    if not os.path.exists(path):
        path = Path(path)
        path.mkdir(parents=True)

if __name__ == "__main__":
    arg= getArgs()