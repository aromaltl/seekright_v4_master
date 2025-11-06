import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import re
import os
import cv2
import utilites
# from global_variables import variable_obj
import config
import geo
import time
import subprocess

arguments = config.getArgs()
def decdeg2dms(dd):
    is_positive = dd >= 0
    dd = abs(dd)
    minutes, seconds = divmod(dd * 3600, 60)
    degrees, minutes = divmod(minutes, 60)
    degrees = degrees if is_positive else -degrees
    return f"{str(int(degrees)).zfill(2)}.{str(int(minutes)).zfill(2)}.{seconds:05.2f}"


def convert_position_format(df):
    for i in range(df.shape[0]):
        lat, lon = df.iloc[i]["Position"].split("E")
        new_pos = f"N{decdeg2dms(float(lat[1:]))}E{decdeg2dms(float(lon))}"
        df.at[i, "Position"] = new_pos

    return df


def split(ltng):
    '''
    arguments: combined string format of latlong
    returns: latitude and longitude (float)
    '''
    if 'E' in ltng:
        v=ltng.split("E")
        lt=float(v[0][1:])
        ln=float(v[1][:])
    else:
        v=ltng.split("W")
        lt=float(v[0][1:])
        ln=-float(v[1][:])
    if 'S' in ltng:
        lt=-lt
    return lt,ln 
    
def merge(lt,ln):
    '''
    arguments latitude and longitude (float)
    returns combined string format of latlong
    '''
    s=''
    lt=round(lt,6)
    ln=round(ln,6)
    if lt >0:
        s=s+'N'+str(abs(lt))
    else:
        s=s+'S'+str(abs(lt))
    if ln >0:
        s=s+'E'+str(abs(ln))
    else:
        s=s+'W'+str(abs(ln))
    return s

if not os.path.exists("Image-ExifTool-12.22/exiftool"):
    os.system("wget takeleap.in/ml-gui/Image-ExifTool-12.22.zip && unzip Image-ExifTool-12.22.zip && chmod -R 777 .")

def get_gps(video_path):
    try:
        if os.path.exists(video_path.replace(".MP4",".csv")):
            print("exists")
            return pd.read_csv(video_path.replace(".MP4",".csv")),True
        
        tool_path = 'Image-ExifTool-12.22/exiftool'
        # os.system('{} -ee -G3 "{}" > metadata.txt'.format(tool_path, video_path))
        command = f'{tool_path} -ee -G3 "{video_path}" '
        # command = 'Image-ExifTool-12.22/exiftool'
        result = subprocess.run(command,shell=True, stdout=subprocess.PIPE, text=True)
        # f = open('metadata.txt', 'r')
        # meta = f.read().split('\n')
        meta = result.stdout.split('\n')
        # print(meta)
        i = 0
        visited = [meta[0][:8], '']
        val = {'time': [], 'gps': [], 'speed': [], 'diff': []}

        try:
            vname = os.path.basename(video_path).split(".")[0]
            # vdate = datetime(int(vname[:4]), int(vname[4:6]), int(vname[6:8]), int(vname[8:10]), int(vname[10:12]),
            #                  int(vname[12:14]))
            cap = cv2.VideoCapture(video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
        except Exception as ex:
            print(ex)
            return 'Video Error', False

        ############################ Global data ################
        # variable_obj.video_path = video_path
        # variable_obj.video_name = vname
        # variable_obj.fps = fps
        # variable_obj.video_frame_count = total_frames
        #########################################################
        noerror_id = 0
        for x in range(len(meta)):
            if meta[x][:8] not in visited:
                visited.append(meta[x][:8])
                # print(meta[x][:8])
                try:
                    c_t = datetime.strptime(meta[x][-20:-1], "%Y:%m:%d %H:%M:%S") #+ timedelta(hours=5, minutes=30)
                except Exception as ex:
                    print(ex)
                    c_t = datetime.now()
                    noerror_id += 1
                    if len(val['time']):
                        c_t = val['time'][-1]
                    # continue

                if len(val['time']):
                    val['diff'].append(int((c_t - val['time'][-1]).total_seconds()))

                val['time'].append(c_t)
                current = ''
                for ii in range(1, 3):
                    xx = meta[x + ii].split(": ")[1]
                    tem = re.split('[( deg )(\')"]', xx)
                    c = str(round(float(tem[0]) + float(tem[5]) / 60 + float(tem[7]) / 3600, 6))
                    c += '0' * (9 - len(c))
                    current += xx[-1] + c

                val['gps'].append(current)
                val['speed'].append(int(float(meta[x + 3].split(': ')[1])))
        val['diff'].append(0)
        
        df = pd.DataFrame(val)

        val = {'time': [], 'gps': [], 'speed': []}
        for x in range(len(df)):
            val['time'].append(df['time'][x])
            val['gps'].append(df['gps'][x])
            val['speed'].append(df['speed'][x])

        final_df = {'Frame': [], 'Position': [], "Speed": []}

        required_sec = int(np.ceil(total_frames / fps))
        sec_available = len(val['gps'])
        print('required_sec :', required_sec, ', sec_available :', sec_available)
        # compensating for no gps part in the video
        val['gps'] = ['N0.00000E0.00000'] *max(0,required_sec-sec_available  ) + val['gps']
        val['speed'] = [0] *max(0,required_sec-sec_available  ) + val['speed']
        nnn=len(val['gps'])-1

        ############################################################

        pos=[val['gps'][0]]
        cc=0
        ## interpolate gps seconds which is having same values using next one
        for x in range(1,len(val['gps'])):
            distance_from_center =geo.calculateDistance('24.7236846, 44.982107',val['gps'][x])
            if 'N0.0000' in val['gps'][x] or 'E0.0000' in val['gps'][x] or distance_from_center > 1250 :
                val['gps'][x]='N0.00000E0.00000'


            if (pos[-1]==val['gps'][x] or val['gps'][x]=='N0.00000E0.00000') and pos[-1]!='N0.00000E0.00000':
                cc+=1   
            else:
                if cc:
                    lat1,lon1=split(pos[-1])
                    lat2,lon2=split(val['gps'][x])
                    lats=np.linspace(lat1,lat2,cc+2)[1:-1]
                    lons=np.linspace(lon1,lon2,cc+2)[1:-1]
                    for lat,lon in zip(lats,lons):
                        pos.append(merge(lat,lon))
                pos.append(val['gps'][x])
                cc=0

        pos=pos+[pos[-1]]*cc
        val['gps']=pos
        ############################################################

        for x in range(0, total_frames, arguments['frames_skipped']):
            
            
            final_df['Frame'].append(x)
            # To avoid out of index error
            x=int(min(nnn,x/fps))
            final_df['Position'].append(val['gps'][x])
            final_df['Speed'].append(val['speed'][x])
    
        final_df = pd.DataFrame(final_df)
        if video_path[-4:] == ".MP4":

            final_df.to_csv(video_path.replace('.MP4', '.csv')) 
        return final_df, True
    except Exception as ex:
        print(ex)
        utilites.PrintException()
        return 'Gps Error', False


if __name__ == '__main__':
    import glob
    import time
    for x in glob.glob('/home/sultan/Downloads/Depth/*.MP4',recursive=True):
        print(x)
        t=time.time()
        get_gps(x)
        print(time.time()-t)


        