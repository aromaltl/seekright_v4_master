import json
import os
from uuid import uuid4
import xml.etree.ElementTree as ET
import glob
import cv2


def check(a):
    try:
        int(a)
        return True
    except:
        return False
################################################################
all_videos = glob.glob("/run/user/1000/gvfs/smb-share:server=anton.local,share=roadis_phase4/ml_support/Qatar oct-2025/Green Zone/SH-MAP GREEN ( 12 &13-Oct-2025)/Green - Main Road - Doha Road, Al Khor, Qatar (12-Oct-2025)/**/*.MP4",recursive=True)
all_jsons = glob.glob("/run/user/1000/gvfs/smb-share:server=anton.local,share=roadis_phase4/ml_support/Qatar oct-2025/Green Zone/SH-MAP GREEN ( 12 &13-Oct-2025)/Green - Main Road - Doha Road, Al Khor, Qatar (12-Oct-2025)/**/*sam_anno.json",recursive=True)
site_name = "Qatar"
date = '2025-11-06'
version = '001'
splitvalue = 3000
writeimg = False

################################################################
def normal(a,w,h):
    # print(a,"@#@#")
    # cont = []
    for x in a:
        x[0]=(x[0]/w)*100
        x[1]=(x[1]/h)*100
    return a
    

def getvideo(x,allvideos):
    for y in allvideos:
        if "_".join(x.split("_")[:3]) in y:
            return y 
    return None

def create_image_url(filepath):
    """
    Label Studio requires image URLs, so this defines the mapping from filesystem to URLs
    if you use ./serve_local_files.sh <my-images-dir>, the image URLs are localhost:8081/filename.png
    Otherwise you can build links like /data/upload/filename.png to refer to the files
    """
    filename = os.path.basename(filepath)
    return f'/data/local-files/?d={labelstudio_path}/images/{filename}'

months = date.split("-")[:2]
months.append(date)
date = months

print(date)
data_path = os.path.join("/run/user/1000/gvfs/smb-share:server=anton.local,share=labelstudio/data", date[0],date[1],date[2], version)
os.makedirs(f"{data_path}/images" ,exist_ok=True)
images_folder_path = os.path.join(data_path, "images")
labels_folder_path = os.path.join(data_path, "labels")
labelstudio_path = data_path[data_path.find("labelstudio") + len("labelstudio") + 1:]
empty_xml_count = 0



json_data = []
classnames = set()
for json_p in all_jsons: 
    # json_p = '/home/tl028/Downloads/saudi_annotate/2024_0728_104453_F_sam_anno.json'
    with open(json_p,'r') as f:
        data = eval(f.read())


    error_count = 0
    vname = os.path.basename(json_p)

    video = getvideo(vname,all_videos)
    vname = vname.split(".")[0]
    cap = cv2.VideoCapture(video)
    ret,fr = cap.read()
    print(json_p)
    for x in data:
        if check(x):
            #print(x)

            try:
                results = []
                # GOT = False
                for ast in data[x]:
                    # if 'hevro' in ast:
                        # GOT =True
                    for vals in data[x][ast]:
                        region_id = str(uuid4())[:10]
                        label_name=ast
                        contours=vals[3]
                        classnames.add(ast)
           

                        result = {
                            "id": region_id,
                            "from_name": "polygon",
                            "to_name": "image",
                            "original_width": fr.shape[1],
                            "original_height": fr.shape[0],
                            "image_rotation": 0,
                            "value": {
                                "points" : normal(contours[0],fr.shape[1],fr.shape[0]),
                                "polygonlabels" : [label_name],
                                "closed" : True
                            },
                            'score': 0.811,
                            "type": "polygonlabels",
                        }
                        results.append(result)
                image_file_path = f"{vname}_{x}.jpeg"
                # image_name+=1
                if  results :
                    if writeimg:
                        cap.set(1,int(x))
                        ret,fr = cap.read()
                        print("saved",f"{data_path}/images/{image_file_path}")
                        cv2.imwrite(f"{data_path}/images/{image_file_path}",fr)
                    json_data.append({
                            'data': {
                                'image': create_image_url(image_file_path),
                                'site_name' : site_name 
                            },
                            'predictions': [{
                                'model_version': "sam_2",
                                'score': 0.811,
                                'result': results,
                            }]
                    })
                else:
                    # json_data.append({
                    #         'data': {
                    #             'image': create_image_url(image_file_path)
                    #         }
                    # })
                    empty_xml_count += 1
                # cv2.imwrite(f"{data_path}/images/{image_file_path}.jpeg",fr)
            # except ET.ParseError:
            #     empty_xml_count += 1
            #     json_data.append({
            #             'data': {
            #                 'image': create_image_url(image_file_path),
            #                 'site_name' : site_name 
            #             }
            #     })
            except Exception as e:
            
                print(e)
                # json_data.append({
                #         'data': {
                #             'image': create_image_url(image_file_path),
                #             'site_name' : site_name 
                #         }
                # })
                error_count += 1
print(f"total {len(json_data)}")
json_data = json_data[::5]
for i in range(0,len(json_data),splitvalue):
    print(f"saving import{i}.json")
    with open( f"import{i}.json", "w") as json_file:
        json.dump(json_data[i:i+splitvalue], json_file)
for x in classnames:
    print(x)

