import torch
from depth_anything_v2.dpt import DepthAnythingV2
import cv2
import numpy as np
import time
import os
from PIL import Image   

class DepthAnything:

    depth_anything_model = None

    def __init__(self, ):
        
        self.device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
        self.encoder = 'vitl'  # Options ['vitb','vits','vitl']
        self.input_size = 518
    


        if DepthAnything.depth_anything_model is None:
            DepthAnything.depth_anything_model = DepthAnythingV2(self.encoder)
            os.makedirs('checkpoints',exist_ok=True)
            if not  os.path.exists('./checkpoints/depth_anything_v2_vitl.pth'):
                os.system("wget -P checkpoints/ 'https://seekright.takeleap.in/SeekRight/checkpoints/depth_anything_v2_vitl.pth'")

            DepthAnything.depth_anything_model.load_state_dict(torch.load(f'./checkpoints/depth_anything_v2_{self.encoder}.pth', map_location='cuda'))
            DepthAnything.depth_anything_model = DepthAnything.depth_anything_model.to(self.device).eval()



    def get_depth(self,image=None, coordinate_x=None, coordinate_y=None):

        # raw_image = cv2.imread(image) 
        raw_image = image     
        depth = DepthAnything.depth_anything_model.infer_image(raw_image,self.input_size)
        depth = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth = depth.astype(np.uint8)
        depth = depth[coordinate_y, coordinate_x]

        # meters =  round(-0.09683*depth + 18.27)
        if depth < 70:
            meters = max(4,round(55.48 -2.161*depth + 0.03*depth*depth - 0.0001262 *depth**3))
        else:
            meters = 3
        meters = min(20,meters)
        # print(meters,depth,"@@@@")
        return meters
    

    
    # def get_the_meters(self,image=None, coordinate_x=None, coordinate_y=None):

    #     depth = self.get_depth(image, coordinate_x, coordinate_y)
        
    #     # scaling_factor = self.scaling_factor()
    #     meter = round(-0.09683*depth + 21.27)
    #     print(depth,meter,"@@@@")
    #     return meter 
    



# image='/home/sultan/103_256_33_5004.976_155_8902.jpeg'
# coordinate_x=1480
# coordinate_y= 1080
# # [2244, 1220], [1480, 1080]
# # coordinate_x=2244
# # coordinate_y= 1220
# # 2244, 1220
# # 1480, 1080

# # coordinate_x=951
# # coordinate_y=1369

# start_time = time.time()
# depth = DepthAnything()

# meter = depth.get_the_meters(image, coordinate_x, coordinate_y)
# print('meter', meter)
# end_time = time.time()
# time_taken = end_time - start_time
# print('time_taken',time_taken)
