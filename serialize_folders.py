import os
import tqdm
import re

def rename_videos(root_folder):
    # Walk through each directory and subdirectory starting from root_folder
    for root, _, files in os.walk(root_folder):
       
        serial_number = 1
        

        files.sort()
        for filename in files:
            # print(filename,root)
            if filename.endswith(('_F.MP4', )):
                

                new_filename = filename.replace("_F.MP4",f"_{serial_number:04}F.MP4")
                # new_filename = re.sub("_[0-9]{4}F.MP4","_F.MP4",filename)
                
                # Construct the full path of the current file
                current_file = os.path.join(root, filename)
                new_file = os.path.join(root, new_filename)
                
                # Rename the file
                os.rename(current_file, new_file)
                # print(current_file, new_file)
                # Increment the serial number for the next file
                serial_number += 1

if __name__ == "__main__":
    root_folder = '/run/user/1000/gvfs/smb-share:server=enigma.local,share=saudi_video_sync/From Applus Office(21-7-2024)/'
    rename_videos(root_folder)
