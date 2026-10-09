import os
import copy
import numpy as np
import json
from imutils.video import FPS
import imutils
import cv2
import torch
import torchvision.ops.boxes as bops

folders = [x for x in os.listdir("RESULT")]


def get_box(cords, shape):
    cords = np.array(cords) * np.array(shape * 2)
    cords[:2] -= cords[2:] / 2
    cords[2:] += cords[:2]
    return [int(cord) for cord in np.round(cords, 0)]


def tracker_get_box(cords, shape):
    cords = np.array(cords) * np.array(shape * 2)
    cords[:2] -= cords[2:] / 2
    return tuple(np.round(cords, 0).astype(int))


def get_box_rev(cords, shape):
    cords = np.array(cords).astype(float)
    cords[2:] -= cords[:2]
    cords[:2] += cords[2:] / 2
    cords = np.array(cords) / np.array(shape * 2)
    return tuple(cords)


def check_overlap(cord1, cord2, gap_counter):
    box1_ = torch.tensor([cord1], dtype=torch.float)
    box2_ = torch.tensor([cord2], dtype=torch.float)
    iou_ = bops.box_iou(box1_, box2_)
    return iou_ > (0.5 if gap_counter == -1 else 0.65 if gap_counter == 0 else 0.1)


def pop_entry(dictionary, key_, value_):
    dictionary[key_].remove(value_)
    if len(dictionary[key_]) == 0:
        del dictionary[key_]


def get_direction(frames_, cords_at):
    frame_itr = 0
    frame_ = frames_[frame_itr]
    frame_stack = {}
    main_cord = cords_at[frame_][0]
    if len(cords_at[frame_]) > 1:
        frame_stack[frame_] = cords_at[frame_][1:]
    frame_ += 1
    frame_itr += 1
    final_cords = [main_cord]

    counter_ = 10
    while counter_ != 0 :
        flag_ = False
        if frame_ in frame_stack:
            for cord in frame_stack[frame_]:
                if check_overlap(main_cord, cord, -1):
                    pop_entry(frame_stack, frame_, cord)
                    # flag_change = False
                    main_cord = cord
                    final_cords.append(main_cord)
                    flag_ = True
                    break
        elif frame_ in cords_at:
            stack_itr = 0
            stack_lst = []
            cords_lst = cords_at[frame_]
            while stack_itr != len(cords_lst):
                coord_ = cords_lst[stack_itr]
                if check_overlap(main_cord, coord_, -1):
                    main_cord = coord_
                    final_cords.append(main_cord)
                    flag_ = True
                    break
                else:
                    stack_lst.append(coord_)
                stack_itr += 1
            if flag_:
                stack_lst.extend(cords_lst[stack_itr + 1:])
            if len(stack_lst) != 0:
                frame_stack[frame_] = stack_lst

        if flag_:
            counter_ -= 1
        else:
            counter_ = 10
            stack_keys = list(frame_stack.keys())
            if len(stack_keys) != 0:
                frame_ = stack_keys[0]
                main_cord = frame_stack[frame_][0]
                frame_stack[frame_].remove(main_cord)
                if len(frame_stack[frame_]) == 0:
                    del frame_stack[frame_]
            else:
                frame_ = frames_[frame_itr]
                frame_stack = {}
                main_cord = cords_at[frame_][0]
                if len(cords_at[frame_]) > 1:
                    frame_stack[frame_] = cords_at[frame_][1:]
                frame_itr += 1
            final_cords = [main_cord]
        frame_ += 1
        if frame_itr == len(frames_):
            break

    x1, y1, x2, y2 = np.diff(np.array(final_cords), axis=0).mean(axis=0)
    rate_x = x1 if abs(x1) > abs(x2) else x2
    return 1 if rate_x < 0 else -1


for folder in folders:
    img = folder[-4:]
    txt_path = "not_noisy/"+folder+"/"+img
    vid_path = "RESULT/"+folder+"/IMG_"+img
    out_path = "CVoutput/IMG_"+img+".txt"
    if os.path.exists(vid_path+".m4v"):
        vid_path += ".m4v"
    else:
        vid_path += ".MOV"

    print(img, vid_path, txt_path)
    vs = cv2.VideoCapture(vid_path)
    frame = vs.read()[1]
    # frame = frame[1] if args.get("video", False) else frame
    frame = imutils.resize(frame, width=1500)
    frame_shape = frame.shape[1::-1]

    frames_dict = {}
    count = 0
    tracker_parameter = {}
    tent_id = 0

    frames = []
    for file_n in os.listdir(txt_path):
        frames.append(int(file_n[9:-4]))
        with open(txt_path + '\\' + file_n) as file:
            frames_dict[int(file_n[9:-4])] = sorted([get_box([float(val) for val in
                                                              line.strip().split(" ")[1:]], frame_shape)
                                                     for line in file.readlines()], key=lambda a: a[0])
    frames = sorted(frames)
    #print(frames)

    length = 0
    box = None
    counter = 0
    x = frame_shape[0] / 2
    w = 100
    x_direc = get_direction(frames, copy.deepcopy(frames_dict))
    #print(x_direc)
    # flag_change = False

    while len(frames_dict) != 0 and length != len(frames_dict):
        tent_id += 1
        length = len(frames_dict)
        # flag_change = True
        count += 1
        res = list(frames_dict.keys())[0]
        frame_box = get_box_rev(frames_dict[res][0], frame_shape)
        # pop_entry(frames_dict, res, frames_dict[res][0])
        tracker = cv2.legacy.TrackerBoosting_create()
        initBB = None  # initializing bb

        vs = cv2.VideoCapture(vid_path)
        fps = None
        frames = 0

        frame_num = int(res)
        flag = False
        coord = []
        tracker_dict = {}
        flaggy = True

        while True:
            frame = vs.read()[1]
            # frame = frame[1] if args.get("video", False) else frame
            if frame is None:
                break
            frame = imutils.resize(frame, width=1500)
            (H, W) = frame.shape[:2]

            if initBB is not None:
                if not flag and ((x_direc == 1 and x < 0) or (x_direc == -1 and x + w > frame_shape[0])):
                    flag = True
                    counter = 10
                if flag and counter == 0:
                    break
                elif flag:
                    counter -= 1

                if flaggy:
                    (success, box) = tracker.update(frame)
                    flaggy = False
                else:
                    old_box = box
                    (success, box) = tracker.update(frame)
                    box = (*box[:-2], old_box[-2] + 1, old_box[-1] + 1)

                if success:

                    # frame_count = int(vs.get(cv2.CAP_PROP_POS_FRAMES))
                    (x, y, w, h) = [int(v) for v in box]
                    tracker_dict[frames] = [x, y, x + w, y + h]
                    if frames in tracker_parameter.keys():
                        tracker_parameter[frames].append([[tent_id], [x, y, w, h]])
                    else:
                        tracker_parameter[frames] = [[[tent_id], [x, y, w, h]]]
                    # print((x, y), (x + w, y + h))
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                fps.update()
                fps.stop()
                info = [
                    ("Tracker", "kcf"),
                    ("Success", "Yes" if success else "No"),
                    ("FPS", "{:.2f}".format(fps.fps())),
                ]
                for (i, (k, v)) in enumerate(info):
                    text = "{}: {}".format(k, v)
                    cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            #cv2.imshow("Frame", frame)
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

        for key, value in tracker_dict.items():
            if key in frames_dict:
                coord = frames_dict[key]
                removable_list = []
                for i in coord:
                    box1 = torch.tensor([i], dtype=torch.float)
                    box2 = torch.tensor([value], dtype=torch.float)
                    iou = bops.box_iou(box1, box2)

                    if (not flag and iou > 0.3) or (flag and iou > 0):
                        removable_list.append(i)

                for i in removable_list:
                    frames_dict[key].remove(i)
                    if len(frames_dict[key]) == 0:
                        del frames_dict[key]
                        # flag_change = False

        # print(frames_dict)

    print(count - 1)

    final_out = ""
    data = json.loads(json.dumps(tracker_parameter))
    for frame in data:
        for coords in data[frame]:
            final_out += frame + " " + str(coords[0][0]) + " " + " ".join([str(_) for _ in coords[1]]) + "\n"

    with open(out_path, "w") as file:
        file.write(final_out)
