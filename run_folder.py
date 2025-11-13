import os
import glob
import pandas as pd
import argparse
import datetime
import numpy as np
import requests
import time
import re
import subprocess ,sys

def split_list(videos,id,share=1,):
    tot=len(videos)
    chunk=int(np.ceil(tot/share))
    return videos[id*chunk:(id+1)*chunk]


if __name__ == "__main__":
    wait=0.3
    while True:
        try:
            r = requests.get("http://10.20.30.8:5000/video")
            x=eval(r.text)["video"]
        except requests.exceptions.ConnectionError :
            wait=min(600,wait+3)
            print(f"no connection sleeping for {wait} sec")
            time.sleep(wait)
            continue
        wait=0.3
        print(x)
        if len(x)==0:
            print("No video")
            break
        if 'F.MP4' in x and "Out" not in x:
            # vehicle_name = [x for x in video_path.split('/')][-2]
            # vehicle_name = os.path.dirname(x).split("/")[-1]
            vehicle_name = re.findall('/[0-9]{3}/', x)

            if vehicle_name is not None and len(vehicle_name):
                v = "vehicle_"+vehicle_name[0][1:-1]
            else:
                v = "NA"
            vehicle_name=v
            video_name = os.path.basename(x)
            vname=video_name.replace("_","").replace("F.MP4",f"_{vehicle_name}.MP4")
            # os.system(f"python3 v4.py -i '{x}' --vehicle '{vehicle_name}' --vname '{vname}' ")
            cmd = [
                "python3", "v4.py",
                "-i", x,
                "--vehicle", vehicle_name,
                "--vname", vname
            ]
            try:
                result = subprocess.run(
                    cmd,
                    timeout=900,          # e.g. 300 seconds = 5 minutes
                    check=True,           # raises CalledProcessError if exit != 0
                    text=True,            # decode output as text
                    # capture_output=True   # capture stdout/stderr
                    stdout=sys.stdout,
                    stderr=sys.stderr
                )
                #print("✅ Success:", result.stdout)

            except subprocess.TimeoutExpired:
                print(f"⏱️ Timeout: {x} took too long, skipping...")

            except subprocess.CalledProcessError as e:
                print(f"❌ Error running {x}: {e.stderr}")
                
                print("done")
    print("All Done !!!!")


# if __name__ == "__main__":



#     ################################### 
#     idd=""
#     share=""
#     path = "/mnt/share/**/*.MP4"
#     process,rename = 1,1
#     ###################################



#     if process==1:
#         paths=[x for x in glob.glob(path,recursive=True) if 'utput' not in x]
#         paths.sort()
#         paths=split_list(paths,idd,share)


        

#         # x = datetime.datetime.now()
#         for x in paths:

#             print(x)
#             if '_F.MP4' in x and "Out" not in x:
#                 # vehicle_name = [x for x in video_path.split('/')][-2]
#                 vehicle_name = os.path.dirname(x).split("/")[-1]
#                 if len(vehicle_name):
#                     v = "vehicle_"+vehicle_name[1:]
#                 else:
#                     v = "NA"
#                 vehicle_name=v
#                 video_name = os.path.basename(x)
#                 vname=video_name.replace("_","").replace("F.MP4",f"_{vehicle_name}.MP4")
#                 os.system(f"python3 v4.py -i '{x}' --vehicle '{vehicle_name}' --vname '{vname}' ")
            
#                 print("done")
#         print("All Done !!!!")
#     # if rename==1:
#     #     for video_path in glob.glob(path,recursive=True):
#     #         print(video_path)
#     #         if '_F.' in video_path and "Out" not in video_path:
#     #             # vehicle_name = [x for x in video_path.split('/')][-2]
#     #             vehicle_name = os.path.dirname(video_path).split("/")[-1]
#     #             if len(vehicle_name):
#     #                 v = "vehicle_"+vehicle_name[1:]
#     #             else:
#     #                 v = "NA"
#     #             vehicle_name=v

#     #             video_name = os.path.basename(video_path)
#     #             directory = os.path.dirname(video_path)
#     #             # os.rename(video_path,os.path.join(directory,video_name.replace("_","").replace("F.MP4",f"_{vehicle_name}.MP4")))
#     #     print("renamed")