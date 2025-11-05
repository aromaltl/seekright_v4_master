import json
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import subprocess
import cv2
import glob
from sam2.build_sam import build_sam2_video_predictor
import requests
import time
from depth import DepthAnything
linear = set(["Bad_Concrete_Barrier","Concrete_Barrier",
        "Bad_Crash_Barrier","Bad_Fence", "Steel_Barrier",
        "Bad_Guard_Rails", "Bad_Steel_Barrier",
        "Bad_Jersey_Barrier","Bad_Kerbs",
        "Bad_Lane","Bad_MBCB",
        "Patch","Crash_Barrier",
        "Fence",
        "Guard_Rails",
        "Jersey_Barrier",
        "Kerbs",
        "MBCB",
        "Lane",
        "Pot_Holes",
        "Cracks",
        "Manhole",
        "Left_Road_Markings",
        "Right_Road_Markings",
        "Sand_Accumulation","flag"])

def mask_to_bounding_box(mask):

    
    y_nonzero, x_nonzero = torch.where(mask  )

    if len(y_nonzero) == 0 or len(x_nonzero) == 0:

        return None,None,None
    
    y_min = y_nonzero.min().item()
    y_max = y_nonzero.max().item()
    x_min = x_nonzero.min().item()
    x_max = x_nonzero.max().item()


    # return  [[x_min, y_min], [x_max, y_max]] 
    mid = y_nonzero.shape[0]//2
    return  [[x_min, y_min], [x_max, y_max]] , x_nonzero[mid].item(), y_nonzero[mid].item()

def converting_to_asset_format(data_json, total_frames):


    data = {}
    print("converting_to_asset_format started...")
    for i in range(0, total_frames, 2):
        i = str(i)
        # print(i)
        if i not in data_json:
            continue

        for asset in data_json[i]:
            for index, v in enumerate(data_json[i][asset]):
                if asset not in data:
                    data[asset] = {}
                if asset in linear and  int(v[0]) < 9000: # ignore linear which arenot manually added
                    continue
                if int(v[0]) not in data[asset]:
                    data[asset][int(v[0])] = []
                data[asset][int(v[0])].append([i, v[1], v[2]])
    print("converting_to_asset_format ended!!")
    return data
class near:
    def __init__(self,):
        sam2_checkpoint = "./checkpoints/sam2_hiera_large.pt"
        model_cfg = "sam2_hiera_l.yaml"
        if not os.path.exists(sam2_checkpoint ):
            os.system("wget -P checkpoints/ https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt ")


        self.data =None
        self.predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint,device = torch.device("cuda"))
        self.dpt = DepthAnything()

    def set_folder(self,video,frame_no,temp_path='./temp'):

        os.makedirs(temp_path,exist_ok=True)
        try:
            os.system(f"rm  {temp_path}/*")
        except:
            pass
        # print(video,"###@@@@@@@@@@@@@@@######&@@@@@!!!!!!@@@@@@@@@@@")
        cap = cv2.VideoCapture(video)
        cap.set(1,int(frame_no))
        i=-1

        for i in range(25):
            ret, frame = cap.read()
            cap.read()
            
            if not ret :
                break
            cv2.imwrite(f"{temp_path}/{i:05d}.jpeg",frame)
            
        cap.release()
        


    def find_near(self,val,data,asset,frame,asset_for_dpt,temp_path="./temp"):
        
        inference_state = self.predictor.init_state(video_path=temp_path)
        self.predictor.reset_state(inference_state)

        box = np.array(val[1:3], dtype=np.float32).reshape(-1)
        _,out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=1,
            box = box,
        )
        breaker = 0
        frame-=2
        for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):

            # if out_frame_idx>0:
            frame+=2
            mask = out_mask_logits[0][0]>0.0
            bbox,xm,ym = mask_to_bounding_box(mask)
            h,w = mask.shape
            # if bbox[0][0] <5 or bbox[0][1] <5 or bbox[1][0] > w-5 or bbox[1][1] > h -5 : 
                
            if bbox is None:
                breaker+=1
                if breaker>4:
                    break
                continue
            breaker = 1
            if 1 :#bbox[0][0] > 5 and bbox[0][1] >5 and bbox[1][0] < w-5 and bbox[1][1] < h -5 : 
                if out_frame_idx>0:   
                    if str(frame) not in data:
                        data[str(frame)]={}
                    if asset not in data[str(frame)]:
                        data[str(frame)][asset]=[]

                    data[str(frame)][asset].append([val[0],bbox[0],bbox[1]])
                # print(bbox)
                asset_for_dpt.setdefault(asset, {})[val[0]] = (frame,xm,ym)
            
            
    def find_add(self,video,path):
        last_data = {}
        try:
            with open(path,"r") as f:
                # data = json.loads(data)
                data = eval(f.read())
            asset_for_dpt = {}#defaultdict(lambda: defaultdict(list))
        except Exception as ex:
            print(ex,"corrupt json")
            return 

        for frame, value in data.items():
            try:
                frame=int(frame)
            except :
                continue
            for ast,v in value.items():

                for ids in v:

                    if ast not in last_data:
                        last_data[ast]={}
                    
                    last_data[ast][ids[0]]=(ids,frame)
        # print(last_data)
        for assets in last_data:
            if assets in linear or "start" in assets.lower() or "end"  in assets.lower():
                continue
            for ids in last_data[assets]:
                ids,frame = last_data[assets][ids]
                temp_path = os.path.join('temp',os.path.basename(video).replace(".MP4",""))
                self.set_folder(video,frame,temp_path = temp_path)
                self.find_near(ids,data,assets,frame,asset_for_dpt,temp_path=temp_path)
                print("########################")
        # print(data)
        with open(path.replace("_annotation.json","_sam.json"),"w") as f:
            json.dump(data,f)

        final_json = {"Assets":[]}
        cap = cv2.VideoCapture(video)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(3))
        height = int(cap.get(4))

        data = converting_to_asset_format(data,total_frames=total_frames)

        for asset in data:
            if asset in linear:
                continue
            for ids in data[asset]:
                if len(data[asset][ids])==0:
                    continue
                val = data[asset][ids][0]
                for ijk in data[asset][ids][::-1]:
            
                    if ijk[1][0]>5 and ijk[1][1]>5 and ijk[2][0]< width-5 and ijk[2][1] < height-5:
                        val=ijk
                        # print("#@#@#####@#@#@##")
                        break
                
                last = data[asset][ids][-1]

                if (val[1][0] + val[2][0]) / 2 > width / 2:
                    Asset = "RIGHT_" + asset
                else:
                    Asset = "LEFT_" + asset

                # name , id, frame , x1y1 , x2y2,[], frame_new
                # push_meters = 12 if 'ight' in asset else 5
                # print(asset,ids,asset_for_dpt)
                # print(asset_for_dpt,asset,ids,"##@#")
                fr , xm, ym = asset_for_dpt.get(asset,{}).get(str(ids),(None,None,None))
                if fr is not None and 'ight' not in Asset:
                    cap.set(1,fr)
                    
                    try:
                        frame = cap.read()[1]
                        push_meters = self.dpt.get_depth(frame,xm,ym)
                    except  Exception as ex:
                        print(ex)
                        push_meters = 8
                    # cv2.circle(frame,(xm,ym), 10, (0,255,0), -1)
                    # cv2.imshow('depth',frame[::3,::3])
                    # cv2.waitKey(0)
                else:
                    push_meters = 8
                final_json["Assets"].append([Asset, int(ids), int(val[0]), val[1], val[2], ['', ''], push_meters, int(last[0])])
        
        cap.release()
        with open(path.replace("_annotation.json","_final.json"),"w") as f:
            json.dump(final_json,f)
    
    # def update_final_json(self,video,path):
    #     try:
    #         with open(path,"r") as f:
    #             data = eval(f.read())
    #     except Exception as ex:
    #         print(ex)
    #         return
    #     linear = set( [
    #             "Bad_Crash_Barrier","Bad_Fence",
    #             "Bad_Guard_Rails",
    #             "Bad_Jersey_Barrier","Bad_Kerbs",
    #             "Bad_Lane","Bad_MBCB",
    #             "Patch","Crash_Barrier",
    #             "Fence",
    #             "Guard_Rails",
    #             "Jersey_Barrier",
    #             "Kerbs",
    #             "MBCB",
    #             "Lane",
    #             "Pot_Holes",
    #             "Cracks",
    #             "Manhole",
    #             "Left_Road_Markings",
    #             "Right_Road_Markings",
    #             "Sand_Accumulation"])
    #     for x in data["Assets"]:
          
    #         if "start" in x[0].lower() or "end"  in x[0].lower():
    #             continue
    #         if x[0].replace("RIGHT_","").replace("LEFT_","") in  linear:
    #             continue
    #         # if int(x[1])>9000:
    #         #     continue
    #         frame = int(x[2])
    #         self.set_folder(video,frame)
    #         try:
    #             inference_state = self.predictor.init_state(video_path="./temp")
    #         except Exception as ex:
    #             continue
    #         self.predictor.reset_state(inference_state)

    #         box = np.array(x[3:5], dtype=np.float32).reshape(-1)

    #         _,out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
    #             inference_state=inference_state,
    #             frame_idx=0,
    #             obj_id=1,
    #             box = box,
    #         )
    #         out_frame_idx=0
    #         valss =[(x[3:5],0)]
    #         breaker =0
    #         for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):

    #             if out_frame_idx>0:

    #                 mask = out_mask_logits[0][0]>0.0
    #                 bbox = mask_to_bounding_box(mask)
    #                 if bbox is None:
    #                     bbox=[[0,0],[0,0]]
    #                     bbox =None
    #                 # print(bbox)
    #                 if bbox is None:# or bbox[1][0]>2558 or bbox[0][0]<1 or bbox[0][1]<1 or bbox[1][1]>1438:
    #                     breaker+=1
    #                     if breaker > 4:
    #                         break
    #                     continue
                    
                    
    #                 valss.append((bbox,out_frame_idx*2))
    #                 breaker=0
    #         if len(valss)>3:
    #             bb,fr =valss[-1]
    #         else:
    #             bb,fr =valss[-1]
            
    #         print(x,"##")
    #         x[2]=x[2]+fr
    #         x[7]=x[7]+fr
    #         x[3:5]=bb
    #         print(x,"###")
    #     print(data,"@@@@##")
    #     if '_final.json' in path:
    #         path = path.replace("_final.","_final_new.")
    #     with open(path,"w") as f:
    #         json.dump(data,f)
            



if __name__ == "__main__":



    obj = near()
    wait=0.5
    # obj.find_add('/mnt/Backup/share/new/2024_0805_121806_0037F.MP4','/mnt/Backup/share/new/2024_0805_121806_0037F/2024_0805_121806_0037F_annotation.json')

    # exit()


    while 1:
        try:
            r = requests.get("http://10.20.30.8:5050/processed")
            v=eval(r.text)
            print(v,"@@@@@@@@###########@@@@@@@@@@ values")
            jsn = v["json"]
            vid = v["video"]
            # obj.update_final_json(vid,jsn)
            obj.find_add(vid,jsn)
            wait=0.3
            if "/mnt/share" in vid:
                os.system(f"rm '{vid}'")
        # except requests.exceptions.ConnectionError :
        except Exception as ex:
            print(ex)
            wait+=1
            print(f"no connection sleeping for {min(wait,2)} sec")
            time.sleep(min(wait,2))
            if wait>360:
                break
            continue

            


        

    # video = "/mnt/Backup/share/new/21/101/13-07-2024/2024_0713_084327_0001F.MP4"
    # js ="/mnt/Backup/share/new/21/101/13-07-2024/2024_0713_084327_0001F/2024_0713_084327_0001F_final.json"
    # # # obj.find_add(video,js)
    # # jso ="/home/sultan/Desktop/Seekright-Tool-for-Easy-Master-Audit/Upload_Images/2024_0709_095446_0009F/2024_0709_095446_0009F_final.json"
    # obj.update_final_json(video,js)
    # vid=list(glob.glob("/mnt/Backup/share/new/21/101/13-07-2024/*.MP4"))
    # vid.sort()
    # for x in vid:
    #     print(x)
    #     vname =os.path.basename(x).replace(".MP4","")
    #     js = os.path.join(os.path.dirname(x),vname,vname+"_final.json")
    #     # print(js)
    #     if os.path.exists(js):
    #         video=x
    #         obj.update_final_json(video,js)




    # vid=list(glob.glob("/run/user/1000/gvfs/smb-share:server=enigma.local,share=saudi_video_sync/**/*.MP4",recursive=True))
    # jsn=list(glob.glob("/mnt/Backup/share/jsonss/June/**/*_final.json",recursive=True))
    # jsn.sort()
    # jsons_dict = { os.path.basename(x).replace(".MP4","_final.json"):x for x in vid }
    # for file in jsn:
    #     if "/108/" not in file:
    #         continue

    #     print(file)
    #     name = os.path.basename(file)
    #     vid_path = jsons_dict.get(name,None)
    #     if os.path.exists(file.replace("_final.","_final_new.")):
    #         continue

    #     if vid_path is not None and os.path.exists(vid_path) :
    #         # print(vid_path,file)
    #         obj.update_final_json(vid_path,file)


    # vid=list(glob.glob("/run/user/1000/gvfs/smb-share:server=enigma.local,share=saudi_video_sync/**/*.MP4",recursive=True))
    # jsn=list(glob.glob("/home/sultan/Downloads/Riyaadh_jsons/105_json/*_final.json",recursive=True))
    # jsn.sort()
    # jsons_dict = { os.path.basename(x).replace(".MP4","_final.json"):x for x in vid }
    # for file in jsn:
    #     # if "/110/" not in file:
    #     #     continue
    #     print(file)
    #     name = os.path.basename(file)
    #     vid_path = jsons_dict.get(name,None)
    #     # if os.path.exists(file.replace("_final.","_final_new.")):
    #     #     continue

    #     if vid_path is not None and os.path.exists(vid_path) :
    #         print(vid_path,file)
    #         # print(vid_path,file)
    #         # obj.update_final_json(vid_path,file)

# import os
# import glob
# from flask import Flask,jsonify
# import threading

# # paths=["/mnt/share/From Applus Office(14-07-24)/**/*.MP4",]
# paths=["/mnt/share/new/**/*.MP4"]

# def unprocessed_videos(paths):
# 	videos=[]
# 	for p in paths:
# 		videos+=[x for x in glob.glob(p,recursive=True) if ('utput' not in x and 'F.MP4' in x)]
# 	print(videos)
# 	videos.sort()
# 	temp=[]
# 	for x in videos:
# 		vname = os.path.basename(x).replace(".MP4","")
# 		dir = os.path.dirname(x)
# 		json_name = vname+'_annotation.json'
# 		json_path = os.path.join(dir,vname,json_name)
# 		if not os.path.exists(json_path):
# 			temp.append(x)
# 	print("total unprocessed: ",len(temp))
# 	return temp



# app = Flask(__name__)
# videos=unprocessed_videos(paths)

# lock = threading.Lock()
# @app.route('/reset', methods=['GET'])
# def reset():
# 	global videos
# 	with lock:
# 		videos=unprocessed_videos(paths)
# 	return jsonify({"message":len(videos)}), 200

# @app.route('/video', methods=['GET'])
# def get_string():
# 	global videos
# 	with lock:
# 		total_unprocessed_videos=len(videos)

# 		if total_unprocessed_videos:
# 			response = {'video': videos.pop(0)}
# 			print("remaining: ",total_unprocessed_videos-1)
# 			return jsonify(response), 200
# 		else:

# 			response = {'video': ""}
# 			print("remaining: ",0," reset and check!!!!")
# 			return jsonify(response), 400

# if __name__ == '__main__':
#     app.run(debug=True,host='0.0.0.0')
