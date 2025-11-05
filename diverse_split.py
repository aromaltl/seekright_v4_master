import os
import random
import shutil
import tqdm
import time

def diverse(base_folder = None, train_percent = 80,classes_names=None):
    folder = base_folder
    ind=0
    from collections import Counter
    txt_file_count = 0
    txt_files = []
    for folder,sub,files in os.walk(folder):
        for file in files:
            if file.lower().endswith('.txt') and "classes.txt" not in file:
                    txt_files.append(os.path.join(folder,file))
                    txt_file_count = txt_file_count + 1
    print("txt_file_count: ",txt_file_count)
    class_txt_files = []
    classes_dict = Counter()
    for txt_file in txt_files:
            with open(txt_file,'r') as file:
                lines = file.readlines()
                for line in lines:  
                        parts = line.split()
                        class_id = int(parts[0])
                        classes_dict[class_id] += 1
    # print()
    # print(classes_dict,classes_names)
    for class_id, count in classes_dict.items():

        print(f"Class {class_id} {classes_names[class_id]}: {count} occurrences")
    sorted_classes = dict(sorted(classes_dict.items(), key=lambda x: x[1]))
    sort =sorted_classes.keys()
    sorted_lists = list(sort)
    print(sorted_lists)


    sorted_classes = sorted_lists
    processed_txt = []
    for classes in sorted_classes:
        class_txt_files = []
        for txt_file in txt_files:
            if txt_file not in processed_txt:
                with open(txt_file,'r') as file:
                    lines = file.readlines()
                    for line in lines:
                        parts = line.split()
                        if len(parts)==0:
                             continue
                        class_id = int(parts[0])
                        if classes==class_id:
                            processed_txt.append(txt_file)
                            if txt_file not in class_txt_files:
                                class_txt_files.append(txt_file)
                                break
        
        random.shuffle(class_txt_files)
        total_files = len(class_txt_files)
        print(f'length of class id {classes} is : {total_files}')
        
        train_count = max(1, int((train_percent / 100) * total_files))
        train_set = class_txt_files[:train_count]
        test_set = class_txt_files[train_count:]
        print(f'train count of class id {classes} is : {len(train_set)}')
        print(f'test count of class id {classes} is : {len(test_set)}')
        print()
        print()
        # train_folder = './diversed/train'
        # test_folder = './diversed/test'

        os.makedirs('./diversed/labels/train', exist_ok=True)
        os.makedirs('./diversed/images/train', exist_ok=True)
        os.makedirs('./diversed/images/test', exist_ok=True)
        os.makedirs('./diversed/labels/test', exist_ok=True)

        
        for label in tqdm.tqdm(train_set):
            id_ = os.path.basename(label).split(".")[0]
            mfold = label.split("/")[-3]
            img = label.replace("labels/","images/").replace(".txt",".jpeg")

            os.system(f"cp '{img}' './diversed/images/train/{mfold}_{id_}.jpeg' ")
            os.system(f"cp '{label}' './diversed/labels/train/{mfold}_{id_}.txt' ")
            ind+=1


        for label in tqdm.tqdm(test_set):
            id_ = os.path.basename(label).split(".")[0]
            mfold = label.split("/")[-3]

            img = label.replace("labels/","images/").replace(".txt",".jpeg")
            os.system(f"cp '{img}' './diversed/images/test/{mfold}_{id_}.jpeg' ")

            os.system(f"cp '{label}' './diversed/labels/test/{mfold}_{id_}.txt' ")
            ind+=1

        # time.sleep(0.4)

        # for files in train_set:
        #     relative_path = os.path.relpath(files, base_folder)
        #     # print('rel',relative_path)
        #     target_path = os.path.join(train_folder, relative_path)
        #     # print('tar',target_path)
        #     label_path = files.replace("images","labels")
        #     os.makedirs(os.path.dirname(target_path), exist_ok=True)
        #     shutil.move(files, target_path)

        #     target_path_xml = f'{os.path.dirname(target_path)}/labels'
        #     xml_path = f'{os.path.splitext(label_path)[0]}.txt'
        #     if os.path.exists(xml_path):
        #         # print(xml_path)
        #         shutil.move(xml_path,target_path_xml)

        #     target_path_img = f'{os.path.dirname(target_path)}/images'
        #     extensions = ['.jpeg','.jpg','.png','.JPEG']
        #     # extensions = ['.jpeg','.jpg','.png','.JPEG']
        #     for extension in extensions:
        #         img_path = f'{os.path.splitext(files)[0]}{extension}'
        #         if os.path.exists(img_path):
        #             # print(img_path)
        #             shutil.move(img_path,target_path_img)
                

        # for files in test_set:
        #     relative_path = os.path.relpath(files, base_folder)
        #     target_path = os.path.join(test_folder, relative_path)
        #     label_path = files.replace("images","labels")
        #     os.makedirs(os.path.dirname(target_path), exist_ok=True)
        #     shutil.move(files, target_path)

        #     target_path_xml = f'{os.path.dirname(target_path)}/labels'
        #     xml_path = f'{os.path.splitext(label_path)[0]}.txt'
        #     if os.path.exists(xml_path):
        #         # print(xml_path)
        #         shutil.move(xml_path,target_path_xml)

        #     target_path_img = f'{os.path.dirname(target_path)}/images'
        #     extensions = ['.jpeg','.jpg','.png','.JPEG']
        #     for extension in extensions:
        #         img_path = f'{os.path.splitext(files)[0]}{extension}'
        #         if os.path.exists(img_path):
        #             # print(img_path)
        #             shutil.move(img_path,target_path_img) 
