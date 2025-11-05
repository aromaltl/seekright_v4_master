import os
import glob
from flask import Flask,jsonify
import threading
import re
import json
import time

videos = []


# vid=list(glob.glob("/run/user/1000/gvfs/smb-share:server=enigma.local,share=saudi_video_sync/**/*.MP4",recursive=True))
# jsn=list(glob.glob("/mnt/share/jsonn/**/*annotation.json",recursive=True))

def refresh_queue():
    global videos

    vid=list(glob.glob("/mnt/share/new/**/*.MP4",recursive=True))
    jsn=list(glob.glob("/mnt/share/new/**/*_annotation.json",recursive=True))

    jsn.sort()

    jsons_dict={}
    for x in vid:
        p=os.path.basename(x).replace(".MP4","_annotation.json")
        print(p)
        if p not in jsons_dict:
            jsons_dict[p]=[x]
        else:
            jsons_dict[p].append(x)


   
    for file in jsn:

        name = os.path.basename(file)
        vid_paths = jsons_dict.get(name,None)
        vehicle_name = re.findall('/[0-9]{3}/', file)
        if vehicle_name is  None or len(vehicle_name)==0:
            continue
        vehicle_name=vehicle_name[0]
        
        # if int(vehicle_name[-3:-1]) <4:
        #     continue

        if vid_paths is not None:
            vid_path =vid_paths[0]

            for sim_vid in vid_paths:
                if vehicle_name in sim_vid:
                    vid_path=sim_vid
        else:
            continue

        if os.path.exists(file.replace("_annotation.json","_sam.json")):
            continue

        if vid_path is not None and os.path.exists(vid_path) :

            videos.append((file,vid_path))

    print("Total videos : ",len(videos))
    with open('../data_vid.json', 'w') as file:
        json.dump(videos, file, indent=2)
total_unprocessed_videos=len(videos)
app = Flask(__name__)

lock = threading.Lock()
QUIT =False
@app.route('/processed', methods=['GET'])
def get_string():
    global videos, QUIT
    with lock:
        total_unprocessed_videos=len(videos)
        if  QUIT:
            response = {'video': ""}
            print("Quitting")
            return jsonify(response), 400

        
        if not total_unprocessed_videos:
            
            time.sleep(60)
            refresh_queue()
            total_unprocessed_videos=len(videos)



        if total_unprocessed_videos:
            pathss =videos.pop(0)
            os.system(f"cp '{pathss[1]}' /mnt/Backup/share/temp")
            new_path = os.path.join("/mnt/share/temp",os.path.basename(pathss[1]))
            response = {'video':new_path, 'json':pathss[0] }
            print("remaining: ",total_unprocessed_videos-1)
            return jsonify(response), 200
        else:
            QUIT=True


            response = {'video': ""}
            print("remaining: ",0," reset and check!!!!")
            return jsonify(response), 400

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0',port =5050)

