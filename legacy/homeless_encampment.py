import numbers
import os
import numpy as np
import json
from imutils.video import FPS
import argparse
import imutils
import cv2
import torch
import torchvision.ops.boxes as bops

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="kcf", help="OpenCV object tracker type")
args = vars(ap.parse_args())

vs = cv2.VideoCapture(args["video"])
frame = vs.read()
frame = frame[1] if args.get("video", False) else frame
frame = imutils.resize(frame, width=1500)
frame_shape = frame.shape[1::-1]

directory = "C:\\Users\\HP\\Desktop\\IMSC\\CODE\\RESULT\\IMG_0108\\0108"
frames_dict = {}
count = 0
tracker_parameter = {}
tent_id = 0

def get_box(cords, shape):
    cords = np.array(cords) * np.array(shape*2)
    cords[:2] -= cords[2:]/2
    cords[2:] += cords[:2]
    return [int(cord) for cord in np.round(cords, 0) ]

def tracker_get_box(cords, shape):
    cords = np.array(cords) * np.array(shape*2)
    cords[:2] -= cords[2:]/2
    return tuple(np.round(cords, 0).astype(int))

def get_box_rev(cords, shape):
    cords = np.array(cords).astype(float)
    cords[2:] -= cords[:2]
    cords[:2] += cords[2:]/2
    cords = np.array(cords) / np.array(shape*2)
    return tuple(cords)


for file_n in os.listdir(directory):     
    with open("C:\\Users\\HP\\Desktop\\IMSC\\CODE\\RESULT\\IMG_0108\\0108\\"+file_n) as file:
            frames_dict[int(file_n[9:-4])] = sorted([get_box([float(val) for val in line.strip().split(" ")[1:]], frame_shape) for line in file.readlines()], key=lambda a: a[0])

length = 0

while(len(frames_dict)!=0 and length!=len(frames_dict)):
    tent_id+=1
    length = len(frames_dict)
    count += 1
    res = list(frames_dict.keys())[0]
    frame_box = get_box_rev(frames_dict[res][0],  frame_shape)

    tracker = cv2.TrackerKCF_create()
    initBB = None                           # initializing bb

    vs = cv2.VideoCapture(args["video"])
    fps = None
    frames = 0

    frame_num = int(res)
    flag = False
    coord = []
    tracker_dict = {}
    flaggy = True

    while True:        
        frame = vs.read()
        frame = frame[1] if args.get("video", False) else frame
        if frame is None:
            break
        frame = imutils.resize(frame, width=1500)
        (H, W) = frame.shape[:2]

        if initBB is not None:
            if not flag and x<0:
                flag = True
                counter = 10
            if flag and counter == 0:
                break
            elif flag:
                counter-=1

            if flaggy:
                (success, box) = tracker.update(frame)
                flaggy = False
            else:
                old_box = box
                (success, box) = tracker.update(frame)
                box = (*box[:-2], old_box[-2]+1, old_box[-1]+1)

            if success:
            
                #frame_count = int(vs.get(cv2.CAP_PROP_POS_FRAMES))
                (x, y, w, h) = [int(v) for v in box]
                tracker_dict[frames] = [x, y, x + w, y + h]
                if frames in tracker_parameter.keys():
                    tracker_parameter[frames].append([[tent_id], [x, y, w, h]])
                else:
                    tracker_parameter[frames] = [[[tent_id], [x, y, w, h]]]
                #print((x, y), (x + w, y + h))
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            fps.update()
            fps.stop()
            info = [
                ("Tracker", args["tracker"]),
                ("Success", "Yes" if success else "No"),
                ("FPS", "{:.2f}".format(fps.fps())),
            ]
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF

        if frames == frame_num:
            initBB = tracker_get_box(frame_box, frame_shape)
            tracker.init(frame, initBB)
            (x, y, w, h) = [int(v) for v in initBB] 
            tracker_dict[frames] = [x, y, x + w, y + h]
            if frames in tracker_parameter.keys():
                    tracker_parameter[frames].append([[tent_id], [x, y, w, h]])
            else:
                tracker_parameter[frames] = [[[tent_id], [x, y, w, h]]]            
            fps = FPS().start()

        # if the `q` key was pressed, break from the loop
        elif key == ord("q"):
            break
        frames += 1        
    
        

    vs.release()
    cv2.destroyAllWindows()

    dict = {}
    for key, value in tracker_dict.items():
        if key in frames_dict:
            coord = frames_dict[key]
            removable_list = []
            for i in coord:
                box1 = torch.tensor([i], dtype = torch.float)
                box2 = torch.tensor([value], dtype = torch.float)
                iou = bops.box_iou(box1, box2)
                
                if (not flag and iou > 0.3) or (flag and iou > 0):
                    removable_list.append(i)                    
            
            for i in removable_list:
                frames_dict[key].remove(i)                    
                if len(frames_dict[key]) == 0:
                    del frames_dict[key]

    print(frames_dict)

with open('tracker_parameter.json', 'w') as f:
    json.dump(tracker_parameter, f)

print(count-1)