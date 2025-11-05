import os
import shutil
from PIL import Image
import imagehash
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

path = '/mnt/enigma/my-data/2024/06/2024-06-09/001/images' #input path
output_path ='14' #output path, u hav 2 create folder
hash_values = []

for folder, sub, files in os.walk(path):
    # print('files',files)
    for file in files:
        if file.lower().endswith(('.jpg', '.jpeg', '.JPEG','.png')):
            file_path = os.path.join(folder, file)
            # print('fil path',file_path)
            relative_path = os.path.relpath(file_path, path)
            hash_value = imagehash.average_hash(Image.open(file_path))

            if hash_value not in hash_values:
                hash_values.append(hash_value)
            else:
                relative_output_path = os.path.join(output_path, os.path.dirname(relative_path))
                os.makedirs(relative_output_path, exist_ok=True)     
                destination_path = os.path.join(relative_output_path, file)
                shutil.move(file_path, destination_path)
                # print(f'{file_path} is moved into similar images directory {destination_path}')

                xml_path = f'{os.path.splitext(file_path)[0]}.xml'
                if os.path.exists(xml_path):
                    shutil.move(xml_path, relative_output_path)
                    # print(f'{xml_path} is moved into similar images directory {relative_output_path}')

                txt_path = f'{os.path.splitext(file_path)[0]}.txt'
                if os.path.exists(txt_path):
                    shutil.move(txt_path, relative_output_path)
                    # print(f'{txt_path} is moved into similar images directory {relative_output_path}')
