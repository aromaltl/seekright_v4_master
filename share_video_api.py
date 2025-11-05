import os
import glob
from flask import Flask,jsonify
import threading

path="/mnt/share/**/*.MP4"

def unprocessed_videos(path):
	videos=[x for x in glob.glob(path,recursive=True) if ('utput' not in x and '_F.MP4' in x)]
	videos.sort()
	temp=[]
	for x in videos:
		vname = os.path.basename(x).replace(".MP4","")
		dir = os.path.dirname(x)
		json_name = vname+'_annotation.json'
		json_path = os.path.join(dir,vname,json_name)
		if not os.path.exists(json_path):
			temp.append(x)
	print("total unprocessed: ",len(temp))
	return temp



app = Flask(__name__)
videos=unprocessed_videos(path)

lock = threading.Lock()
@app.route('/reset', methods=['GET'])
def reset():
	global videos
	with lock:
		videos=unprocessed_videos(path)
	return jsonify({"message":len(videos)}), 200

@app.route('/video', methods=['GET'])
def get_string():
	global videos
	with lock:
		total_unprocessed_videos=len(videos)

		if total_unprocessed_videos:
			response = {'video': videos.pop(0)}
			print("remaining: ",total_unprocessed_videos-1)
			return jsonify(response), 200
		else:

			response = {'video': ""}
			print("remaining: ",0," reset and check!!!!")
			return jsonify(response), 400

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0')

