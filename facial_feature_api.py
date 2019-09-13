from flask import Flask, render_template
from keras.preprocessing import image
from keras.models import load_model
import os
import numpy as np
import cv2,dlib
import time
from multiprocessing import Process, Value
import json

model1 = load_model('new_data/model_mustache_final.h5')
model2 = load_model('new_data/model_eyeglass_final.h5')
model3 = load_model('new_data/model_gender.h5')
emotions1 = ('moustache', 'nonmoustache')
emotions2 = ('eyeglass', 'noeyeglass')
emotions3 = ('Male', 'Female')
pts1 = np.float32([[65,80],[150,80],[65,250],[150,250]])
pts2 = np.float32([[0,0],[180,0],[0,320],[180,320]])
global arr_pred1, arr_pred2, arr_pred3
arr_pred1 = []
arr_pred2 = []
arr_pred3 = []
face_area = []
index = []

def dothis(avg_pred1,avg_pred2,avg_pred3):
    global x,y,w,h,face_area,index
    hog_face_detector = dlib.get_frontal_face_detector()
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 500)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 500)
    while(True):
        ret, img = cap.read()
        img = cv2.flip( img, 1 )
        #M = cv2.getPerspectiveTransform(pts1,pts2)
        #img = cv2.warpPerspective(img,M,(320,180))
        faces_hog = hog_face_detector(img, 1)
        face_area = []
        for face in faces_hog:
            try:
                x = face.left()
                y = face.top()
                w = face.right() - x
                h = face.bottom() - y
                face_area.append(w*h)
            except:
                print("Exception 1")
        try:
            index = face_area.index(max(face_area))
            x = faces_hog[index].left()
            y = faces_hog[index].top()
            w = faces_hog[index].right() - x
            h = faces_hog[index].bottom() - y
        except:
            print("Exception face area")
        try:
            if face_area != []:
                cv2.rectangle(img, (x-int(20),y-int(20)), (x+int(1.2*w),y+int(1.2*h)), (0,255,0), 2)
                roi1 = img[y+int(0.5*h):y+h, x:x+w]
                detected_face1 = cv2.resize(roi1, (64, 64))
                img_pixels1 = image.img_to_array(detected_face1)
                img_pixels1 = np.expand_dims(img_pixels1, axis = 0)
                img_pixels1 /= 255
                roi2 = img[y:y+int(0.7*h), x:x+w]
                detected_face2 = cv2.resize(roi2, (64, 64))
                img_pixels2 = image.img_to_array(detected_face2)
                img_pixels2 = np.expand_dims(img_pixels2, axis = 0)
                img_pixels2 /= 255
                roi3 = img[y-int(20):y+int(1.2*h) , x-int(20):x+int(1.2*w)]
                detected_face3 = cv2.resize(roi3, (64, 64))
                img_pixels3 = image.img_to_array(detected_face3)
                img_pixels3 = np.expand_dims(img_pixels3, axis = 0)
                img_pixels3 /= 255
                predictions1 = model1.predict(img_pixels1)
                predictions2 = model2.predict(img_pixels2)
                predictions3 = model3.predict(img_pixels3)
                predictions1 = predictions1[0,0]
                predictions2 = predictions2[0,0]
                predictions3 = predictions3[0,0]
                arr_pred1.append(predictions1)
                arr_pred2.append(predictions2)
                arr_pred3.append(predictions3)
                avg_pred1.value = sum(arr_pred1)/len(arr_pred1)
                #print(value1)
                if avg_pred1.value > 0.60:
                    cv2.putText(img, emotions1[1], (int(x+5), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                else:
                    cv2.putText(img, emotions1[0], (int(x+5), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                if len(arr_pred1) > 25:
                    del arr_pred1[0]
                    
                avg_pred2.value = sum(arr_pred2)/len(arr_pred2)
                if avg_pred2.value > 0.80:
                    cv2.putText(img, emotions2[1], (int(x+15), int(y+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                else:
                    cv2.putText(img, emotions2[0], (int(x+15), int(y+15)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                if len(arr_pred2) > 25:
                    del arr_pred2[0]
                    
                avg_pred3.value = sum(arr_pred3)/len(arr_pred3)
                #print(value3)
                if avg_pred3.value < 0.90:
                    cv2.putText(img, emotions3[1], (int(x+25), int(y+25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
                else:
                    cv2.putText(img, emotions3[0], (int(x+25), int(y+25)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1) 
                if len(arr_pred3) > 25:
                    del arr_pred3[0]
        except:
            print("Exception2")
            
        try:
            cv2.imshow('img',img)
        except:
            print("Exception3")
            
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            cv2.destroyAllWindows()
            
app = Flask(__name__)

@app.route('/')
@app.route('/getData', methods=['GET', 'POST'])
def show_index():
    print(avg_pred1.value)
    return json.dumps({"Mustache": float(avg_pred1.value),
                       "Eyeglass": float(avg_pred2.value),
                       "Gender": float(avg_pred3.value)})

def start():
    global avg_pred1, avg_pred2, avg_pred3
    app.run(host='0.0.0.0', port=5000,use_reloader=False)

if __name__ == "__main__":
   avg_pred1 = Value('f', 0.00)
   avg_pred2 = Value('f', 0.00)
   avg_pred3 = Value('f', 0.00)
   p = Process(target=dothis, args=(avg_pred1, avg_pred2, avg_pred3,))
   p.start()
   start()
   p.join()
