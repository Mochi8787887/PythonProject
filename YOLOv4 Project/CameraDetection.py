# 參考來源: 
# 1. https://data-flair.training/blogs/pedestrian-detection-python-opencv/
# 2. https://ithelp.ithome.com.tw/articles/10270068

# 匯入必要的套件
import numpy as np
import cv2
import os
import imutils
import subprocess
import psutil
import time

# 參數設定
NMS_THRESHOLD=0.3
MIN_CONFIDENCE=0.2

# 行人偵測函式，利用 YOLO 模型
def pedestrian_detection(image, model, layer_name, personidz=0):
	(H, W) = image.shape[:2]
	results = []

    # 將圖片轉換成 YOLO 模型的輸入格式
	blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416),
		swapRB=True, crop=False)
	model.setInput(blob)
	layerOutputs = model.forward(layer_name)

	boxes = []
	centroids = []
	confidences = []

    # 模型輸出的每個檢測框
	for output in layerOutputs:
		for detection in output:

			scores = detection[5:]
			classID = np.argmax(scores)
			confidence = scores[classID]

			if classID == personidz and confidence > MIN_CONFIDENCE:

				box = detection[0:4] * np.array([W, H, W, H])
				(centerX, centerY, width, height) = box.astype("int")

				x = int(centerX - (width / 2))
				y = int(centerY - (height / 2))

				boxes.append([x, y, int(width), int(height)])
				centroids.append((centerX, centerY))
				confidences.append(float(confidence))
	idzs = cv2.dnn.NMSBoxes(boxes, confidences, MIN_CONFIDENCE, NMS_THRESHOLD)
    
    # 若有偵測到目標，則將結果存入 results
	if len(idzs) > 0:
		for i in idzs.flatten():
			(x, y) = (boxes[i][0], boxes[i][1])
			(w, h) = (boxes[i][2], boxes[i][3])
			res = (confidences[i], (x, y, x + w, y + h), centroids[i])
			results.append(res)
	return results

# 讀入類別名稱
labelsPath = "coco.names"
LABELS = open(labelsPath).read().strip().split("\n")

# 讀入yolo-v4參數以及設定檔
weights_path = "yolov4-tiny.weights"
config_path = "yolov4-tiny.cfg"
model = cv2.dnn.readNetFromDarknet(config_path, weights_path)
layer_name = model.getLayerNames()
layer_name = [layer_name[i[0] - 1] for i in model.getUnconnectedOutLayers()]

# 針對筆電視訊影像進行偵測，也可改為其他影像來源
cap = cv2.VideoCapture(0)

# 持續偵測人的迴圈
while True:
    (grabbed, image) = cap.read()

    if not grabbed:
        break
    
    # 調整影像大小
    image = imutils.resize(image, width=700)
    
    # 進行行人偵測
    results = pedestrian_detection(image, model, layer_name,
		personidz=LABELS.index("person"))

    # 畫出偵測到的每個方框
    for res in results:
	    cv2.rectangle(image, (res[1][0],res[1][1]), (res[1][2],res[1][3]), (0, 255, 0), 2)

    # 顯示偵測結果
    cv2.imshow("GGFP",image)
    
    # 若當前影像中有兩個人以上，則關掉當前瀏覽器並開啟指定網站
    if len(results) >= 2:
        subprocess.Popen('taskkill /im chrome.exe')
        time.sleep(0.1)
        subprocess.Popen("start chrome https://learn.microsoft.com/zh-tw/sql/sql-server/?view=sql-server-ver16",shell = True)
        break
        
    # 按下 Esc 鍵結束迴圈
    key = cv2.waitKey(1)
    if key == 27:
        break

# 釋放攝影機資源並關閉視窗
cap.release()
cv2.destroyAllWindows()
