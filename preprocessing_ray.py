import cv2
import numpy as np
import time, os
import sys
import json
import pandas as pd
import ray
from facenet_pytorch import MTCNN
import torch


device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))
sentense_label_dict = {}
mtcnn = MTCNN(keep_all=True, device=device,select_largest=False)

def get_file_list(extension):
    root = os.path.dirname(os.path.abspath(__file__))
    datalist = []
    for path, dirs, files in os.walk(root):
        dir_path = os.path.join(root,path)
        for file in files:
            if os.path.splitext(file)[1] == extension:
                datalist.append(os.path.join(dir_path, file))
    return datalist

def get_lip_data(file_path):
    webcam = cv2.VideoCapture(file_path)
    lip_data = []

    if not webcam.isOpened():
        print("Could not open webcam")
        exit()
    
    while webcam.isOpened():
        status, frame = webcam.read()
        if not status:
            break
        boxes,prob,points=mtcnn.detect(frame,landmarks=True)
    
        frame_draw =frame.copy()
        if boxes is None:
            print("얼굴 인식에 실패하였습니다")
            continue
        boxes=boxes.astype(int)
        points=points.astype(int)

        max_box=prob.argmax()
        ly = (points[0][3][1]+points[0][3][1])//2
        y = boxes[max_box][3]-boxes[max_box][1]
        x = boxes[max_box][2]-boxes[max_box][0]
        
        y_min = ly-y//8
        y_max = ly+y//8
        x_min = points[0][3][0]-x//10
        x_max = points[0][4][0]+x//10
        lip_data.append([x_min,y_min,y_max,x_max])

        
    return lip_data

@ray.remote
def mean_std(videos):
    frame_count = 0
    mean_values = []
    std_values = []
    mean_average=0
    std_average=0

    for video in videos:
        cap = cv2.VideoCapture(video)

        while cap.isOpened():
            ret,frame=cap.read()

            if not ret:
                break

            mean,std=cv2.meanStdDev(frame)
            mean_value=mean.flatten()[0]
            std_value=std.flatten()[0]

            mean_values.append(mean_value)
            std_values.append(std_value)

        mean_average += np.mean(mean_values)
        std_average += np.mean(std_values)
        
    mean_final=mean_average/len(videos)
    std_final=std_average/len(videos)

    return mean_final,std_final



@ray.remote
def preprocessing_video(video,mean,std):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 

    video_count=0
    os.makedirs('dataset', exist_ok=True)
    
    start = time.time()
    print(f"video: {os.path.basename(video)}")

    
    # step 1: 입술 부분만 가져온 뒤 흑백 처리 후 저장
    cap = cv2.VideoCapture(video)
    
    width=cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height=cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    count=cap.get(cv2.CAP_PROP_FRAME_COUNT)
    fps=cap.get(cv2.CAP_PROP_FPS)

    file_name = os.path.basename(video).split('.')[0] + '_preprocessed.mp4'
    saved_path = os.path.join('dataset',file_name)
    out = cv2.VideoWriter(saved_path, fourcc, fps, (96, 96),isColor = False)

    frame_count=0
    lib_data = get_lip_data(video)
    
    while cap.isOpened():
        ret,frame=cap.read()
        if not ret:
            break
        lip=frame.copy()
        lip=lip[lib_data[frame_count][1]:lib_data[frame_count][2],lib_data[frame_count][0]:lib_data[frame_count][3]]
        frame_count+=1

        lip = cv2.resize(lip, dsize=(96,96))
        lip = cv2.cvtColor(lip, cv2.COLOR_BGR2GRAY)
        out.write(lip)
        if cv2.waitKey(1)& 0xFF == ord('q'):
            break
    video_count+=1

    cap.release()
    out.release()
    
    print(f"[Finish {time.time()-start}]: {file_name} ")

def preprocessing(filename):
    ray.init(num_cpus=12)

    # video_list = get_file_list('.webm')
    video_list = [filename + '.webm']
    mean_std_results = mean_std.remote(video_list)
    mean, std = ray.get(mean_std_results)
    mean=round(mean,3)
    std=round(std,3)
    
    # 멀티프로세싱 작업을 위한 풀 생성
    ray_results = [preprocessing_video.remote(video, mean,std) for video in video_list]
    ray.get(ray_results)
    ray.shutdown()
    
if __name__ == '__main__':
    preprocessing("bf2c632e-6641-4ff7-a905-7f8d9a64ea80")