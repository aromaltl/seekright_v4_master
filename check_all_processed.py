import glob
import os
path="/mnt/Backup/share/**/*.MP4"
videos=[os.path.basename(x).replace(".MP4","") for x in glob.glob(path,recursive=True) if 'utput' not in x]
annot = [os.path.basename(x).replace("_annotation.json","") for x in glob.glob(path.replace(".MP4","annotation.json"),recursive=True) ]

for x in set(videos)-set(annot):
	print(x)
print("total videos        : ",len(videos),"",len(set(videos)))
print("total finished      : ",len(annot))
print("total uniq finished : ",len(set(annot)))
print("total unfinished    : ",len(set(videos)-set(annot)))

# videos=[x for x in glob.glob(path,recursive=True) if 'utput' not in x]
# for x in videos:
# 	vname = os.path.basename(x).replace(".MP4","")
# 	dir = os.path.dirname(x)
# 	json_name = vname+'_annotation.json'
# 	json_path = os.path.join(dir,vname,json_name)
# 	print(json_path,x)
# 	if not os.path.exists(json_path):
# 		print(x,"not found")