from imutils.video import FPS
import argparse
import imutils
import cv2
import numpy as np
import json

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, help="path to input video file")
ap.add_argument("-t", "--tracker", type=str, default="csrt", help="OpenCV object tracker type")
args = vars(ap.parse_args())

tracker = cv2.TrackerCSRT_create()
initBB = None                           # initializing bb

vs = cv2.VideoCapture(args["video"])
fps = None
frames = 0
frame_num = 401
#frame_box = (0.985026, 0.531713, 0.0299479, 0.106019 )
#frame_box = (0.951042, 0.565509, 0.0682292, 0.0884259)
frame_box = (0.248, 0.6773428232502966, 0.156, 0.18979833926453143)


def get_box(cords, shape):
    cords = np.array(cords) * np.array(shape*2)
    cords[:2] -= cords[2:]/2
    return tuple(np.round(cords, 0).astype(int))


coord = []
dict = {}
while True:
    frame = vs.read()
    frame = frame[1] if args.get("video", False) else frame
    if frame is None:
        break
    frame = imutils.resize(frame, width=1500)
    (H, W) = frame.shape[:2]

    if initBB is not None:
        if x<0:
            break
        (success, box) = tracker.update(frame)
        if success:
           
            #frame_count = int(vs.get(cv2.CAP_PROP_POS_FRAMES))
            (x, y, w, h) = [int(v) for v in box]
            dict[frames] = [x, y, x + w, y + h]               

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
        #print(frame.shape[1::-1])
        initBB = get_box(frame_box, frame.shape[1::-1])
        tracker.init(frame, initBB)
        (x, y, w, h) = [int(v) for v in initBB] 
        dict[frames] = [x, y, x + w, y + h] 
        fps = FPS().start()


    # if the `q` key was pressed, break from the loop
    elif key == ord("q"):
        break
    frames += 1

with open('tracker.json', 'w', encoding='utf-8') as f:
    json.dump(dict, f, ensure_ascii=False, indent=4)
    
vs.release()
cv2.destroyAllWindows()