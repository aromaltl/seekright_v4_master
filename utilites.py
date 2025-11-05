import mysql.connector
import datetime
import json
import linecache
import os
import sys
import zipfile
import ast
import pandas as pd
import requests

import numpy as np
from config import getArgs

arguments = getArgs()

# from logger_helper import logger_obj
# from db_helper import db_helper_obj
# from global_variables import variable_obj
#from new_compare import find_position
# from fixed_compare import group_coordinates
# from fixed_compare import find_nearest_coordinate

def find_position(data,position):
    test_coors_ = data['Position'].tolist()
    test_coordinates, test_coors = group_coordinates(test_coors_, x=True)
    nearest_point = find_nearest_coordinate(position, test_coordinates)
    return test_coordinates.index(nearest_point)
    
def enter_logger(message):
    pass
    #logger_obj.logger.info(str(datetime.datetime.now()) + " - " + message)


def get_db_connection():
    sql_engine = "mysql://{}:{}@{}/{}".format(arguments["user"], arguments["password"],
                                              arguments["host"], arguments["database"])
    # print(sql_engine)
    engine = create_engine(sql_engine)
    con = engine.connect()
    return con


def get_db_connection2():
    mydb = mysql.connector.connect(
        host=arguments["host"],
        user=arguments["user"],
        password=arguments["password"],
        database=arguments["database"]
    )
    my_cursor = mydb.cursor()
    return my_cursor


def PrintException():  # For printing the exception in required format
    exc_type, exc_obj, tb = sys.exc_info()
    f = tb.tb_frame
    lineno = tb.tb_lineno
    filename = f.f_code.co_filename
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, f.f_globals)
    error_text = str(datetime.datetime.now()) + f' EXCEPTION IN ({filename}, LINE {lineno} "{line.strip()}"): {exc_obj}'
    print(error_text)
    #logger_obj.logger.error(error_text)


def write_to_excel(video_name, data, sheet):
    try:
        excel_path = arguments['csv_path'] + video_name + '.xlsx'
        if not os.path.exists(excel_path):
            data.to_excel(excel_path, sheet_name=sheet, index=False)
        else:
            '''with pd.ExcelWriter(excel_path, engine='openpyxl', mode='a') as writer:
                data.to_excel(writer, sheet_name=sheet, index=False)'''
            with pd.ExcelWriter(excel_path,if_sheet_exists="overlay",mode="a", engine="openpyxl") as writer:
                data.to_excel(writer,sheet_name=sheet, index_label= 'Sr.No')
    except Exception as error:
        print('Error in writing',error)

def save_to_db(data):
    if arguments['save_sql']:
        connection = get_db_connection()
        temp_data = data.copy()
        if "Direction" in temp_data.columns:
            temp_data = temp_data.drop(["Direction"], axis=1)
        if "Distance" in temp_data.columns:
            temp_data = temp_data.drop(["Distance"], axis=1)
        temp_data.to_sql(con=connection, name="anomaly_audit", if_exists='append', index=False)


def remove_video(video_path):
    if arguments["remove_processed_video"]:
        if db_helper_obj.check_future_videos(video_path):
            try:
                os.remove(video_path)
            except Exception as ex:
                print("Error in removing video")
                #logger_obj.logger.warning("Error while removing video")


def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))


def zipit(dir_list, zip_name):
    zipf = zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED)
    for dir in dir_list:
        zipdir(dir, zipf)
    zipf.close()


def dms2dd(value):
    degrees = float(value[1:3])
    minutes = float(value[4:6])
    seconds = float(value[7:12])
    dd = float(degrees) + float(minutes) / 60 + float(seconds) / (60 * 60)
    dd = str(round(dd, 6))
    return dd


def convert_position(pos):
    latitude = dms2dd(pos[:12])
    longitude = dms2dd(pos[12:])
    final_pos = "N" + str(latitude) + "E" + str(longitude)
    return final_pos


def reUploadToAuditPage(video_name):
    images_zip_file_path = f"{arguments['save_image_path']}{variable_obj.algorithm}_{video_name}.zip"
    json_file_path = f"{arguments['save_image_path']}{variable_obj.algorithm}_{video_name}_anomaly.json"
    json_response_path = f"{arguments['video_output_path']}{variable_obj.algorithm}_{video_name}_anomaly_response.json"
    final_json = f"{arguments['video_output_path']}{variable_obj.algorithm}_{video_name}_anomaly_videos_final.json"
    print(images_zip_file_path, json_file_path)
    try:

        if os.path.exists(images_zip_file_path) and os.path.exists(json_file_path):
            images_zip_file = open(images_zip_file_path, "rb")
            f = open(json_file_path, "r")
            # print("Anomaly:",json.loads(f.read()))
            anomalies_li = ast.literal_eval(f.read())
            print("reuploading")
            response = requests.post(f"{arguments['server_ip']}/AuditPage/add_anomaly_audit_page",
                                     files={"file": images_zip_file}, data={"anomalies": json.dumps(anomalies_li)},
                                     verify=False, timeout=120)

            print("Anomaly:", video_name, response.text, "#")
            f.close()
            if response.status_code == 200:
                new_response = response.json()
                if os.path.exists(json_response_path):
                    old_response = json.loads(open(json_response_path, "r").read())
                    if old_response['responseData'][0]['records'] is None:
                        old_response['responseData'][0]['records'] = []

                    if new_response['responseData'][0]['records'] is not None:
                        for resp in old_response['responseData'][0]['records']:
                            if resp not in new_response['responseData'][0]['records']:
                                new_response['responseData'][0]['records'].append(resp)
                    else:
                        new_response = old_response

                with open(json_response_path, 'w') as f:
                    f.write(json.dumps(new_response))
                print("new_json_response", new_response)
                db_helper_obj.anomaly_uploaded()
                os.remove(images_zip_file_path)
                os.remove(json_file_path)
            else:
                db_helper_obj.upload_attempts_update(site_statics_uploaded=0, anomaly_uploaded=1)

        else:
            db_helper_obj.anomaly_uploaded()
            #db_helper_obj.error_message_update(video_id=variable_obj.video_id, error_msg="Re-upload file not found")
            db_helper_obj.error_message_update(video_id=variable_obj.video_id, error_msg="")
    except Exception as ex:
        db_helper_obj.upload_attempts_update(site_statics_uploaded=0, anomaly_uploaded=1)
        PrintException()



def uploadToAuditPage(data, video_name, use_case=False):
    try:
        flag, site_id = db_helper_obj.get_site_id()
        #logger_obj.logger.info("Site Id" + str(site_id))
        lhs_site_id, rhs_site_id = str(site_id).split(",")
        images_zip_file_path = f"{arguments['save_image_path']}{variable_obj.algorithm}_{video_name}.zip"
        json_file_path = f"{arguments['save_image_path']}{variable_obj.algorithm}_{video_name}_anomaly.json"

        json_response_path = f"{arguments['video_output_path']}/{variable_obj.algorithm}_{video_name}_anomaly_response.json"

        anomaly_video_json = f"{arguments['video_output_path']}/{variable_obj.algorithm}_{video_name}_anomaly_videos.json"

        directory_path = arguments['save_image_path']
        folders_to_be_ziped = []
        for folderName, subfolders, filenames in os.walk(directory_path):
            for folder in subfolders:
                if folder.find(video_name) != -1:
                    folders_to_be_ziped.append(folderName + folder)

        zipit(folders_to_be_ziped, images_zip_file_path)
        # print(get_site_id())

        # lhs_site_id = 16z
        # rhs_site_id = 17
        anomalies_li = []
        anomalies_video_li = []
        if "extra_master_image" not in list(data.columns):
            data["extra_master_image"]=[""]*len(data)
        if os.path.exists(json_file_path):

            f = open(json_file_path, "r")
            anomalies_li = ast.literal_eval(f.read())

            f.close()
        if os.path.exists(anomaly_video_json):
            # print("json exist")
            f = open(anomaly_video_json, "r")
            anomalies_video_li = ast.literal_eval(f.read())
            # print("before appending fixed anomaly list")
            # print(anomalies_video_li)
            f.close()

        for i in range(data.shape[0]):
            chainage = str(data.iloc[i]["Chainage"]).replace("-LHS", "").replace("-RHS", "")
            if chainage == "NA":
                continue
            if "Comments" not in data.columns:
                comment = ""
            else:
                comment = data.iloc[i]["Comments"]
            # position = convert_position(data.iloc[i]['Pos'])
            if int(data.iloc[i]["Speed"]) < 4:
                continue
            position = data.iloc[i]['Pos']
            item = {"position": position, "side": str(data.iloc[i]["Side"]), "comment": comment,
                    "asset": str(data.iloc[i]["Asset"]), "direction": data.iloc[i]['Direction'],
                    "site_id": lhs_site_id if data.iloc[i]['Direction'] == "LHS" else rhs_site_id,
                    "master_image": str(data.iloc[i]['Frame_Master']), "chainage": chainage,
                    "test_image": str(data.iloc[i]['Frame_Test']), "video_name": str(data.iloc[i]["video_name"]),
                    "speed": str(data.iloc[i]["Speed"]), "plaza": str(data.iloc[i]["Plaza"]),
                    "extra_image": str(data.iloc[i]["Extra_Image"]),"extra_master_image":str(data.iloc[i]["extra_master_image"]),
                    "algorithm": str(data.iloc[i]["Algorithm"])}
            found=False

            for j in arguments['IGNORE_CHAINAGES']:
                if float(j[0])<=float(item['chainage'])<=float(j[1]) and item['direction'] == str(j[2]):
                    print("droped value",item['chainage'],j[0],j[1],j[2])
                    found=True
                    break
            if item not in anomalies_li and not found:
                anomalies_li.append(item)

            if item not in anomalies_video_li and not found:
                anomalies_video_li.append(item)

        images_zip_file = open(images_zip_file_path, "rb")
        # print()
        # print(images_zip_file_path)
        # print("usecase:",use_case)
        # print("after video anomaly list")
        # print(anomalies_video_li)

        f = open(json_file_path, "w")
        f.write(json.dumps(anomalies_li))
        f.close()
        f = open(anomaly_video_json, "w")
        f.write(json.dumps(anomalies_video_li))
        f.close()
        print(json.dumps(anomalies_li),flush= True)
        response = requests.post(f"{arguments['server_ip']}/AuditPage/add_anomaly_audit_page",
                                 files={"file": images_zip_file}, data={"anomalies": json.dumps(anomalies_li)},
                                 verify=False, timeout=600)
        print("anomaly_upload_response :", response.text)

        if response.status_code == 200:
            if not use_case:
                db_helper_obj.anomaly_uploaded()
            os.remove(images_zip_file_path)
            os.remove(json_file_path)
            new_response = response.json()
            if os.path.exists(json_response_path):
                old_response = json.loads(open(json_response_path, "r").read())
                if old_response['responseData'][0]['records'] is None:
                    old_response['responseData'][0]['records'] = []

                if new_response['responseData'][0]['records'] is not None:
                    for resp in old_response['responseData'][0]['records']:
                        if resp not in new_response['responseData'][0]['records']:
                            new_response['responseData'][0]['records'].append(resp)
                else:
                    new_response = old_response

            with open(json_response_path, 'w') as f:
                f.write(json.dumps(new_response))
            print("new_json_response", new_response, flush=True)
        else:
            if not use_case:
                db_helper_obj.upload_attempts_update(site_statics_uploaded=0, anomaly_uploaded=1)
    except Exception as ex:
        print(f"Exception in upload to audit page function {ex}")
        if not use_case:
            db_helper_obj.upload_attempts_update(site_statics_uploaded=0, anomaly_uploaded=1)
        PrintException()


def uploadRectification(data, video_name, linear=False, night=False):
    try:

        if linear:
            algorithm = "linear:other-linears"
            images_zip_file_path = 'rect_lin_images/' + video_name + ".zip"
            directory_path = 'rect_lin_images/'

        elif night:
            algorithm = "electrical-reflective"
            images_zip_file_path = 'rect_lin_images/' + video_name + ".zip"
            directory_path = 'rect_lin_images/'

        else:
            algorithm = variable_obj.algorithm
            images_zip_file_path = arguments['save_image_path'] + video_name + ".zip"
            directory_path = arguments['save_image_path']

        print("Uploading rectification for: ", video_name, algorithm)
        anomaly_video_json = f"{arguments['video_output_path']}/{algorithm}_{video_name}_anomaly_videos.json"
        json_response_path = f"{arguments['video_output_path']}/{algorithm}_{video_name}_rectification_response.json"



        folders_to_be_ziped = []
        for folderName, subfolders, filenames in os.walk(directory_path):
            for folder in subfolders:
                if folder.find(video_name) != -1:
                    folders_to_be_ziped.append(folderName + folder)

        zipit(folders_to_be_ziped, images_zip_file_path)
        # print(get_site_id())
        flag, site_id = db_helper_obj.get_site_id()
        #logger_obj.logger.info("Site Id" + str(site_id))
        lhs_site_id, rhs_site_id = str(site_id).split(",")
        # lhs_site_id = 16z
        # rhs_site_id = 17
        anomalies_li = []

        video_name = str(video_name)


        for i in range(data.shape[0]):
            chainage = str(data.iloc[i]["Chainage"]).replace("-LHS", "").replace("-RHS", "")
            if chainage == "NA":
                continue
            if "Comments" not in data.columns:
                comment = ""
            else:
                comment = data.iloc[i]["Comments"]
            # position = convert_position(data.iloc[i]['Pos'])
            position = data.iloc[i]['Pos']

            item = {"position": position, "side": str(data.iloc[i]["Side"]), "comment": comment,
                    "asset": str(data.iloc[i]["Asset"]), "direction": data.iloc[i]['Direction'],
                    "site_id": lhs_site_id if data.iloc[i]['Direction'] == "LHS" else rhs_site_id,
                    "master_image": str(data.iloc[i]['Frame_Master']), "chainage": chainage,
                    "test_image": str(data.iloc[i]['Frame_Test']), "video_name": video_name,
                    "speed": str(data.iloc[i]["Speed"]), "plaza": str(data.iloc[i]["Plaza"]),
                    "extra_image": str(data.iloc[i]["Extra_Image"]), "algorithm": str(data.iloc[i]["Algorithm"]),
                    "IsRectified": str(data.iloc[i]["IsRectified"])}
            anomalies_li.append(item)


        images_zip_file = open(images_zip_file_path, "rb")
        print(images_zip_file_path,flush= True)
        print(anomalies_li,flush= True)






        # response = requests.post('http://0.0.0.0:3011/AuditPage/add_anomaly_audit_page',
        #                          files={"file": images_zip_file}, data={"anomalies": json.dumps(anomalies_li)})
        response = requests.post(f"{arguments['server_ip']}/AuditPage/add_anomaly_audit_page",
                                 files={"file": images_zip_file}, data={"anomalies": json.dumps(anomalies_li)},
                                 verify=False)
        print(response.text,flush= True)



        anomalies_video_li = []
        if os.path.exists(anomaly_video_json):
            f = open(anomaly_video_json, "r")
            x = f.read()
            if x == '':
                x = '[]'
            #print('data', x)
            anomalies_video_li = ast.literal_eval(x)
            f.close()
            anomalies_video_li.extend(anomalies_li)


        f = open(anomaly_video_json, "w")
        f.write(json.dumps(anomalies_video_li))
        f.close()

        if response.status_code == 200:

            os.remove(images_zip_file_path)

            new_response = response.json()
            if os.path.exists(json_response_path):
                old_response = json.loads(open(json_response_path, "r").read())
                if old_response['responseData'][0]['records'] is None:
                    old_response['responseData'][0]['records'] = []

                if new_response['responseData'][0]['records'] is not None:
                    for resp in old_response['responseData'][0]['records']:
                        if resp not in new_response['responseData'][0]['records']:
                            new_response['responseData'][0]['records'].append(resp)
                else:
                    new_response = old_response
            with open(json_response_path, 'w') as f:
                f.write(json.dumps(new_response))
            print("rectification_json_response", new_response, json_response_path)

    except Exception as ex:
        print(f"Exception in upload to Rectification page function {ex}")
        PrintException()


def reUploadVideoToAuditPage(video_name, algorithm, video_id):
    print("Started reUploadVideoToAuditPage")
    try:
        video_zip_file_path = f"{arguments['video_output_path']}{algorithm}_{video_name}_anomaly_videos.zip"
        # video_zip_file_path = f"{algorithm}_{video_name}_anomaly_videos.zip"
        video_zip_file_path_usecases = f"{arguments['video_output_path']}{algorithm}_{video_name}_usecases_anomaly_videos.zip"
        json_video_file_path = f"{arguments['video_output_path']}{algorithm}_{video_name}_anomaly_videos.json"
        json_response_path = f"{arguments['video_output_path']}{algorithm}_{video_name}_anomaly_response.json"
        rectification_response_path = f"{arguments['video_output_path']}{algorithm}_{video_name}_rectification_response.json"
        final_json = f"{arguments['video_output_path']}{algorithm}_{video_name}_anomaly_videos_final.json"
        # video_zip_file_path = "/mnt/ML_Drive/StorageDrive1/video_anomaly_test/fixed_20230312122756_000091_anomaly_videos.zip"
        print("json path")
        print(json_response_path)
        if os.path.exists(json_response_path):
            print("Uploading Anomaly videos")
            f = open(json_response_path, "r")
            data = json.loads(f.read())
            print("data", data['responseData'][0]['records'])
            if data['responseData'][0]['records'] == None:
                data['responseData'][0]['records'] = []
            response_json = data
            f.close()
            f = open(json_video_file_path, "r")
            data1 = json.loads(f.read())  # video_upload_json
            final_dic = []
            for i in range(len(data1)):
                for j in range(len(data['responseData'][0]['records'])):
                    # print(data['responseData'][0]['records'][j])
                    new_d = data['responseData'][0]['records'][j]['test_image']
                    # print(new_d, data1[i]['test_image'])
                    if data1[i]['test_image'] == new_d:
                        row_id = data['responseData'][0]['records'][j]['row_id']
                        final_dic.append(
                            {'site_id': data1[i]['site_id'], 'video_name': new_d.replace(".jpeg", ".mp4"),
                             "row_id": row_id})
            print("final_length :", len(data1), len(final_dic))
            json_new = json.dumps(final_dic)

            with open(final_json, "w") as file1:
                file1.write(json_new)
            # json_video_file_path = final_json
            print("final json", final_json)

        else:
            print(json_response_path, "Anomaly video json not exist")

        try:

            if os.path.exists(rectification_response_path):
                print("Uploading Rectification videos")
                f = open(rectification_response_path, "r")
                data = json.loads(f.read())
                response_json = data
                f.close()
                f = open(json_video_file_path, "r")
                data1 = json.loads(f.read())  # video_upload_json
                final_dic = []
                for i in range(len(data1)):
                    for j in range(len(data['responseData'][0]['records'])):
                        new_d = data['responseData'][0]['records'][j]['test_image']
                        # print(new_d, data1[i]['test_image'])
                        if data1[i]['test_image'] == new_d:
                            row_id = data['responseData'][0]['records'][j]['row_id']
                            final_dic.append(
                                {'site_id': data1[i]['site_id'], 'video_name': new_d.replace(".jpeg", ".mp4"),
                                "row_id": row_id, 'IsRectified': 'True'})
                print("final_length :", len(data1), len(final_dic))


                if os.path.exists(final_json):
                    with open(final_json, "r") as file1:
                        already = json.load(file1)
                        final_dic.extend(already)


                json_new = json.dumps(final_dic)
                print(json_new)


                with open(final_json, "w") as file1:
                    file1.write(json_new)
                # json_video_file_path = final_json
                print("final json", final_json)

            else:
                print(rectification_response_path, "Rectofication video json not exist")

        except Exception as ex:
            print("Exception in making rectification video json")
            print(ex)

        print(video_zip_file_path, json_video_file_path)
        try:
            if os.path.exists(video_zip_file_path) and os.path.exists(final_json):
                video_zip_file = open(video_zip_file_path, "rb")
                f = open(final_json, "r")
                video_anomalies = ast.literal_eval(f.read())
                print("reuploading")
                print(video_zip_file)
                print(video_anomalies)
                response = requests.post(f"{arguments['server_ip']}/AuditPage/add_anomaly_video",
                                         files={"file": video_zip_file},
                                         data={"anomalies_video": json.dumps(video_anomalies)},
                                         verify=False, timeout=1000)

                print("video Anomaly:", video_name, response.text, "#")

                if response.status_code == 200:
                    db_helper_obj.video_anomaly_uploaded(video_id)
                    os.remove(video_zip_file_path)
                    # os.remove(final_json)

                else:
                    db_helper_obj.video_upload_attempts_update(video_id, video_anomaly_uploaded=1)
                if os.path.exists(video_zip_file_path_usecases):
                    video_zip_file_usecases = open(video_zip_file_path_usecases, "rb")
                    response = requests.post(f"{arguments['server_ip']}/AuditPage/add_anomaly_video",
                                             files={"file": video_zip_file_usecases},
                                             data={"anomalies_video": json.dumps(video_anomalies)},
                                             verify=False, timeout=1000)

                    print("video Anomaly:", video_name, response.text, "#")

                    if response.status_code == 200:
                        db_helper_obj.video_anomaly_uploaded(video_id)
                        os.remove(video_zip_file_path_usecases)
                        os.remove(final_json)
                    else:
                        db_helper_obj.video_upload_attempts_update(video_id, video_anomaly_uploaded=1)
            else:
                db_helper_obj.video_anomaly_uploaded(video_id)
                db_helper_obj.error_message_update(video_id, error_msg="Anomaly video file not found")
        except Exception as ex:
            print(ex)
            db_helper_obj.video_anomaly_uploaded(video_id)
            db_helper_obj.error_message_update(video_id, error_msg=f"anomaly video upload {ex}")
            PrintException()
    except Exception as ex:
        print(ex)
        db_helper_obj.video_anomaly_uploaded(video_id)
        db_helper_obj.error_message_update(video_id, error_msg=f"Utilities {ex}")
        PrintException()


def uploadToQAAuditPage(data, video_name):
    try:

        images_zip_file_path = arguments['save_image_path'] + video_name + ".zip"
        directory_path = arguments['save_image_path']
        folders_to_be_ziped = []
        for folderName, subfolders, filenames in os.walk(directory_path):
            for folder in subfolders:
                if folder.find(video_name) != -1:
                    folders_to_be_ziped.append(folderName + folder)

        zipit(folders_to_be_ziped, images_zip_file_path)
        # print(get_site_id())
        flag, site_id = db_helper_obj.get_site_id()
        #logger_obj.logger.info("Site Id" + str(site_id))
        lhs_site_id, rhs_site_id = str(site_id).split(",")
        # lhs_site_id = 16z
        # rhs_site_id = 17
        anomalies_li = []
        for i in range(data.shape[0]):
            chainage = str(data.iloc[i]["Chainage"]).replace("-LHS", "").replace("-RHS", "")
            if chainage == "NA":
                continue
            if "Comments" not in data.columns:
                comment = ""
            else:
                comment = data.iloc[i]["Comments"]
            # position = convert_position(data.iloc[i]['Pos'])
            position = data.iloc[i]['Pos']
            item = {"position": position, "side": str(data.iloc[i]["Side"]), "comment": comment,
                    "asset": str(data.iloc[i]["Asset"]), "direction": data.iloc[i]['Direction'],
                    "site_id": lhs_site_id if data.iloc[i]['Direction'] == "LHS" else rhs_site_id,
                    "master_image": str(data.iloc[i]['Frame_Master']), "chainage": chainage,
                    "test_image": str(data.iloc[i]['Frame_Test']), "video_name": str(data.iloc[i]["video_name"]),
                    "speed": str(data.iloc[i]["Speed"]), "plaza": str(data.iloc[i]["Plaza"]),
                    "extra_image": str(data.iloc[i]["Extra_Image"]), "algorithm": str(data.iloc[i]["Algorithm"]),
                    "IsRectified": str(data.iloc[i]["IsRectified"])}
            anomalies_li.append(item)

        images_zip_file = open(images_zip_file_path, "rb")
        print(images_zip_file_path)
        print(anomalies_li)
        # response = requests.post('http://0.0.0.0:3011/AuditPage/add_anomaly_audit_page',
        #                          files={"file": images_zip_file}, data={"anomalies": json.dumps(anomalies_li)})
        response = requests.post(f"{arguments['server_ip']}/AuditPage/add_anomaly_audit_page",
                                 files={"file": images_zip_file}, data={"anomalies": json.dumps(anomalies_li)},
                                 verify=False)
        print(response.text)

        os.remove(images_zip_file_path)
    except Exception as ex:
        print(f"Exception in upload to QA audit page function {ex}")
        PrintException()


def remove_na_rows(data):
    na_rows = []
    for i in range(data.shape[0]):
        if data.iloc[i]['Chainage'].find("NA") != -1:
            na_rows.append(i)
    new_data = data.drop(data.index[na_rows])
    return new_data


def upload_to_dashboard(video_name=None, error_msg=""):
    try:
        if error_msg is None:
            error_msg = ""
        if len(error_msg):
            db_helper_obj.error_message_update(video_id=variable_obj.video_id, error_msg=error_msg)
        if len(db_helper_obj.check_duplicate_data(video_name)) <= 1:
            flag, site_id = db_helper_obj.get_site_id()
            lhs_site_id, rhs_site_id = str(site_id).split(",")
            db_helper_obj.connect_db()
            db_helper_obj.my_cursor.execute(f"select * from video_uploads where id={str(variable_obj.video_id)}")
            video_data = db_helper_obj.my_cursor.fetchall()

            ############################################################################################
            statics_json_path = f"{arguments['save_image_path']}{variable_obj.algorithm}_{video_name.replace('.MP4', '')}_statics.json"

            if os.path.exists(statics_json_path) and 0:
                if video_data[0]['anomaly_uploaded'] != 1:
                    print("Cannot upload tbl statics until anomaly upload complete!!!")
                    return None

                f = open(statics_json_path, "r")
                statics = json.loads(f.read())
                f.close()
                print(statics)
                response = requests.post(f"{arguments['server_ip']}/Master/update_site_statics?site_id",
                                         data={"site_statics": json.dumps([statics])},
                                         verify=False, timeout=1)
                print("update_site_statics response")
                print(response.text)

                if response.status_code == 200:
                    db_helper_obj.site_statics_uploaded()
                    os.remove(statics_json_path)
                    #logger_obj.logger.info("update_site_statics response")
                    #logger_obj.logger.info(response.text)
                return None
            ###########################################################################################
            statics = {"site_id": str(lhs_site_id), 'total_distance_covered': "0",
                       "video_name": video_name, 'chainage_covered': "('0','0')", "day_type": video_data[0]['day_type'],
                       "deleted": str(bool(video_data[0]['is_deleted'])),
                       "idle_time_in_mins": "0",
                       "processing_count": str(video_data[0]["video_processing_count"]),
                       'algorithm': video_data[0]["asset_type"],
                       "is_processed": str(bool(video_data[0]["is_processed"])),
                       "processed_on": str(video_data[0]['processed_on']),
                       "progress_value": str(video_data[0]['progress_value']),
                       'error_type': str(video_data[0]['error_message']),'plaza':str(arguments['site_name'])}
            print(variable_obj.video_path)
            if os.path.exists(variable_obj.video_path.replace('.MP4', '.csv')):


                try:
                    i,j=None,None
                    df = pd.read_csv(variable_obj.video_path.replace('.MP4', '.csv'))
                    print(df)
                    lhs_master_data = pd.read_csv(str(arguments['site_name']) + "/master_data_LHS.csv")
                    df = df[df['Position'] != 'N0.00000E0.00000']
                    df.reset_index(inplace=True,)
                    print(df)
                    for pos in df['Position']:
                        i = find_position(lhs_master_data, pos)
                        if i is not None:
                            break
                    for pos in list(df['Position'])[::-1]:
                        j = find_position(lhs_master_data, pos)
                        if j is not None:
                            break
                    print(i,j)
                except Exception as ex:
                    print(ex)
                    i,j=None,None
                if i is not None and j is not None:
                    if variable_obj.fps is not None:
                        df['Speed'] = df['Speed'].apply(float)
                        idle_frames = np.sum(df['Speed'] < 8)
                        idle_min = int(idle_frames / (variable_obj.fps * 60))
                        statics["idle_time_in_mins"] = str(idle_min)
                    statics["site_id"] = str(rhs_site_id) if i > j else str(lhs_site_id)

                    start_chainage, end_chainage = lhs_master_data['Chainage'][i], lhs_master_data['Chainage'][j]
                    statics['total_distance_covered'] = str(round(abs(start_chainage - end_chainage), 2))
                    statics['chainage_covered'] = str((str(round(start_chainage, 2)), str(round(end_chainage, 2))))
            print(statics)
            f = open(statics_json_path, "w")
            f.write(json.dumps(statics))
            f.close()

            ##########################################
            if video_data[0]['anomaly_uploaded'] != 1:
                print("Cannot upload tbl statics until anomaly upload complete!!!")
                return None
            ##########################################
            response = requests.post(f"{arguments['server_ip']}/Master/update_site_statics?site_id",
                                     data={"site_statics": json.dumps([statics])},
                                     verify=False, timeout=1)
            print("update_site_statics ")
            print(response.text)

            if response.status_code == 200:
                db_helper_obj.site_statics_uploaded()
                #logger_obj.logger.info("update_site_statics response")
                #logger_obj.logger.info(response.text)
            else:
                db_helper_obj.upload_attempts_update(site_statics_uploaded=1, anomaly_uploaded=0)
        else:
            print("Video data is already uploaded")
            db_helper_obj.site_statics_uploaded()
    except Exception as ex:
        db_helper_obj.upload_attempts_update(site_statics_uploaded=1, anomaly_uploaded=0)
        print(f"Exception in upload to dashboard function {ex}")
        PrintException()
