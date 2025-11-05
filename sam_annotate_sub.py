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
from collections import defaultdict

def split(data):
    ndata = []
    n=len(data)
    for x in range(n):
        ndata.append(data[x])
        if "Start" in data[x][0] :
            
            astname_with_side = data[x][0].replace("_Start","")
            for y in range(x,n):
                if astname_with_side+"_End" ==data[y][0]:
                    breaks = ((data[y][2]-data[x][2])//30)
                    for ff in range(1,breaks):
                        ndata.append(data[y][:])
                        ndata[-1][2]=data[x][2]+30*ff-2
                        ndata.append(data[x][:])
                        ndata[-1][2]=data[x][2]+30*ff
                    break

    ndata.sort(key=lambda yy: int(yy[2]))
    with open("temp.json","w") as f:
        json.dump(ndata,f)
    return ndata


linear = set([
        "Bad_Crash_Barrier","Bad_Fence",
        "Bad_Guard_Rails",
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

def mask_to_polygons(mask: np.ndarray) :
        """
        Converts a binary mask to a list of polygons.

        Parameters:
            mask (np.ndarray): A binary mask represented as a 2D NumPy array of
                shape `(H, W)`, where H and W are the height and width of
                the mask, respectively.

        Returns:
            List[np.ndarray]: A list of polygons, where each polygon is represented by a
                NumPy array of shape `(N, 2)`, containing the `x`, `y` coordinates
                of the points. Polygons with fewer points than `MIN_POLYGON_POINT_COUNT = 3`
                are excluded from the output.
        """
        MIN_POLYGON_POINT_COUNT = 2
        contours, _ = cv2.findContours(
            mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contour_area = -1.0
        CONTROL_NUM_POINTS = 0.001 #less value more polygon points
        result=[]
        for contour in contours:
            current_contour_area = cv2.contourArea(contour)
            if contour_area <= current_contour_area:
                contour_area = current_contour_area
                if contour.shape[0] >= MIN_POLYGON_POINT_COUNT:
                    # RDP algorithm to reduce number of polygons
                    rdp_epsilon = CONTROL_NUM_POINTS*cv2.arcLength(contour,True)
                    contour = cv2.approxPolyDP(contour, epsilon = rdp_epsilon, closed=True)
                    c = np.squeeze(contour, axis=1).astype(np.float32)
                    c[:,0] = c[:,0]#/w*100
                    c[:,1] = c[:,1]#/h*100
                    result = [c.tolist()]
        return result


def mask_to_bounding_box(mask):

    
    y_nonzero, x_nonzero = torch.where(mask  )

    if len(y_nonzero) == 0 or len(x_nonzero) == 0:

        return None,None#,None
    # return x_nonzero, y_nonzero
    
    y_min = y_nonzero.min().item()
    y_max = y_nonzero.max().item()
    x_min = x_nonzero.min().item()
    x_max = x_nonzero.max().item()


    return  [[x_min, y_min], [x_max, y_max]] 
    # mid = y_nonzero.shape[0]//2
    # return  [[x_min, y_min], [x_max, y_max]] , x_nonzero[mid].item(), y_nonzero[mid].item()



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
        # cap.set(1,int(frame_no))
        i=-1
        stack=[]
        for i in range(15):
            cap.set(1,int(frame_no)-2*i)
            ret, frame = cap.read()
            # cap.read()
            # print(ret,"####################$$$$$$$$$$$$$$$$$$%%%%%%%%%%%%%%%%%%%%%%",frame_no)
            if not ret :
                break
            # print(i)
            # stack.append(frame)
            # for i,frame in enumerate(stack[::-1]):
            cv2.imwrite(f"{temp_path}/{i:05d}.jpeg",frame)
            
        cap.release()
        


    def update_final_json(self,video,path):
        
        annotations = defaultdict(lambda: defaultdict(list))
        try:
            with open(path,"r") as f:
                data = eval(f.read())
        except Exception as ex:
            print(ex)
            return
        linear = set( [
                "Bad_Crash_Barrier","Bad_Fence",
                "Bad_Guard_Rails",
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
                "Sand_Accumulation"])
        for x in data["Assets"]:
          
            if "start" in x[0].lower() or "end"  in x[0].lower():
                continue
            astname = x[0].replace("RIGHT_","").replace("LEFT_","")
            if astname in  linear:
                continue

            frame = int(x[2])
            cap = cv2.VideoCapture(video)
            self.set_folder(video,frame)
            try:
                inference_state = self.predictor.init_state(video_path="./temp")
            except Exception as ex:
                continue
            self.predictor.reset_state(inference_state)

            box = np.array(x[3:5], dtype=np.float32).reshape(-1)
            # annotations[str(frame)][astname].append(["1",x[3],x[4]])
            _,out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=1,
                box = box,
            )
            out_frame_idx=0
            valss =[(x[3:5],0)]
            breaker =0
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):

                if 1:

                    mask = out_mask_logits[0][0]>0.0
                    bbox = mask_to_bounding_box(mask)
                    mask=mask.detach().cpu().numpy().astype(np.uint8)*255
                    # mask=cv2.resize(mask,(w,h))
                    contours= mask_to_polygons(mask)
                    # contours =''
                    if bbox[0] is None:# or bbox[1][0]>2558 or bbox[0][0]<1 or bbox[0][1]<1 or bbox[1][1]>1438:
                        breaker+=1
                        if breaker > 4:
                            break
                        continue
                    else:
                        # annotations[str(frame-out_frame_idx*2)][astname].append(["1",bbox[0],bbox[1]])
                        annotations[str(frame-out_frame_idx*2)][astname].append(["1",bbox[0],bbox[1],contours])
                        cap.set(1,frame-out_frame_idx*2)
                        ret,fr =cap.read()
                        os.makedirs(f'img/{x[5][1]}',exist_ok=True)
                        cv2.imwrite(f'img/{x[5][1]}/{frame-out_frame_idx*2}.jpeg',fr[bbox[0][1]:bbox[1][1],bbox[0][0]:bbox[1][0]])
                        breaker=0


        path = path.replace("_final.","_sam_anno.")
        with open(path, "w") as f:
            json.dump(annotations,f)
        return annotations
            



if __name__ == "__main__":
    obj = near()
    vid=list(glob.glob("/home/sultan/Downloads/fp_chev/**/*.MP4",recursive=True))
    jsn=list(glob.glob("/home/sultan/Downloads/fp_chev/**/*_final.json",recursive=True))
    jsn.sort()
    jsons_dict = { os.path.basename(x).replace(".MP4","_final.json"):x for x in vid }
    for file in jsn:
        # if "/108/" not in file:
        #     continue

        print(file)
        name = os.path.basename(file)
        vid_path = jsons_dict.get(name,None)
        if os.path.exists(file.replace("_final.","_final_new.")):
            continue

        if vid_path is not None and os.path.exists(vid_path) :
            # print(vid_path,file)
            ann = obj.update_final_json(vid_path,file)



    # obj = near()
    # wait=0.5
    # while 1:
    #     try:
    #         r = requests.get("http://10.20.30.8:5050/processed")
    #         v=eval(r.text)
    #         print(v,"@@@@@@@@###########@@@@@@@@@@ values")
    #         jsn = v["json"]
    #         vid = v["video"]
    #         # obj.update_final_json(vid,jsn)
    #         obj.find_add(vid,jsn)
    #         wait=0.3
    #         if "/mnt/share" in vid:
    #             os.system(f"rm '{vid}'")
    #     # except requests.exceptions.ConnectionError :
    #     except Exception as ex:
    #         print(ex)
    #         wait+=1
    #         print(f"no connection sleeping for {min(wait,2)} sec")
    #         time.sleep(min(wait,2))
    #         if wait>360:
    #             break
    #         continue

            


        

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

