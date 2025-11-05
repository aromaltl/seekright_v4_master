
import glob
import json
import os
import tqdm
import random
import pandas as pd

counter={}
random.seed(2)
# Process each line of the input data
def maptxt(input_data,l_map,g_map,counter,x):
    output_lines = []
    for line in input_data.strip().split('\n'):
        parts = line.split()

        if len(parts)==0 or len(parts[1:]) % 2 !=0 or len(parts[1:]) < 5:

            os.remove(x.replace(".txt",".jpeg").replace("labels","images"))
            os.remove(x)
            print("Error >> ",parts,x)
            break
        # if len(parts[1:]) % 2 !=0 or len(parts[1:]) < 5:
        #     print(parts," Error ",len(parts[1:]))
        class_id = int(parts[0])

        g_id = g_map[l_map[class_id]]
        if l_map[class_id] not in counter:
            counter[l_map[class_id]]=0
        counter[l_map[class_id]]+=1

        output_line = f"{g_id} " + " ".join(parts[1:])
        output_lines.append(output_line)

    # Join the output lines into a single string
    output_data = "\n".join(output_lines)
    return output_data

#########################################
path= "./"
check = 0
##########################################

temp=set()
df=pd.read_csv("http://takeleap.in/ml-gui/common_classes.csv",header=None)
correction = {i:j for i,j in zip(df[0],df[1]) if str(j) not in ("nan","null","NULL")}

for x in glob.glob(os.path.join(path,"**","notes.json"),recursive=True):

    with open(x,"r") as file:
        file= eval(file.read())
        for cat in file["categories"]:
            if cat["name"] in correction:
                temp.add(correction[cat["name"]])
            else:
                
                temp.add(cat["name"])

            # if cat["name"] not in assets_global:
            #     temp.set()
            #     assets_global[cat["name"]]=len(assets_global.keys())

assets_global = {j:i for i,j in enumerate(sorted(list(temp)))}
with open("name.json","w") as f:
    f.write(str(assets_global))

for x in tqdm.tqdm(list(glob.glob(os.path.join(path,"**","notes.json"),recursive=True))):
    with open(x,"r") as file:
        
        file=eval(file.read())
        asset_local = {}
        for t in file["categories"]:
            if t["name"] in correction:

                asset_local[t["id"]] = correction[t["name"]]
            else:

                asset_local[t["id"]] = t["name"]


        for anno in glob.glob(os.path.join(os.path.dirname(x),"labels","*.txt")):
            with open(anno,"r") as f:
                data = maptxt(f.read(),asset_local,assets_global,counter,anno)
            if not check:
                with open(anno,"w") as f:
                    f.write(data)

            # print(data)
    # with open(x,"w") as file:
    #     file.write(assets_global)
for i,j in counter.items():
    print(i,": ",j)
if not check:
    import yaml
    from diverse_split import diverse

    classes_names={i:j for j,i in assets_global.items()}
    diverse(base_folder = path, train_percent = 80,classes_names=classes_names)
    yam = {}
    yam["path"]= os.path.join(os.getcwd(), "diversed")
    yam["train"]="images/train"
    yam["test"]="images/test"
    yam["val"]="images/test"
    yam["names"]=classes_names
    with open(os.path.join(yam["path"],'train_test_data.yml'), 'w') as outfile:
        yaml.dump(yam, outfile, default_flow_style=False)


for x,y in assets_global.items():
    print(f"{y}: {x}")

print("finished  !!!!!")


            



