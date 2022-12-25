import os
import cv2
from models.common import DetectMultiBackend
import torch
import matplotlib.pyplot as plt
from utils.augmentations import letterbox
from utils.dataloaders import IMG_FORMATS, VID_FORMATS
from utils.general import (LOGGER,cv2,non_max_suppression, scale_coords)
import numpy as np
from PIL import Image
from flask import Flask, render_template, request

app = Flask(__name__)
app.secret_key = "secretkey"
@app.route('/')
def welcome():
    return render_template("web.html")

def transform(point, real, new):
    return int(point * new/real)

@app.route("/predict", methods = ["POST"])
def predict():
    global model
    ##
    # WEIGHTS_PATH = (r"/home/vishwesh/Desktop/study/sem7/Project/content/yolov5/best.pt")
    # DEVICE = "cpu"
    # half = False
    # model = DetectMultiBackend(WEIGHTS_PATH, device=DEVICE, dnn=False)
    ##
    stride, names, pt = model.stride, model.names, model.pt
    ##
    conf_thres=0.25  # confidence threshold
    iou_thres=0.45
    classes=None
    save_crop = False
    line_thickness = 3
    ##
    real_classes = ["with helmet","without helmet"]
    ##
    RESIZE = (720, 560)
    COLOR_BORDER = [0, 0, 255]
    COLOR_TEXT = [0, 255, 0]
    save_path = "/home/vishwesh/Desktop/study/sem7/prac/DL/PROJ/static"
    file = request.files.get("img")
    name = file.filename
    if name == "":
        return render_template("error.html")
    img_path = os.path.join(save_path,"raw.jpeg")
    # print(img_path)
    file.save(img_path)
    print("saved")
    file_name = name
    img0 = cv2.imread(img_path)
    img_size = 640
    stride = 32
    auto = True
    img = letterbox(img0, img_size, stride=stride, auto=auto)[0]
    img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    img = np.ascontiguousarray(img)
    im = torch.from_numpy(img).to(DEVICE)
    im = im.float()
    im = im/255
    im = im[None]
    pred = model(im, augment=False, visualize=False)
        # t3 = time_sync()
        # dt[1] += t3 - t2
    # print("hello")

    pred = non_max_suppression(pred, conf_thres, iou_thres, classes, False, max_det=1000)
    # print(pred)
    for i, det in enumerate(pred):
        p, im0 = img_path, img0.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        imc = im0.copy() if save_crop else im0  # for save_crop
        real_width, real_height, _ = im0.shape
        new = cv2.resize(im0, (RESIZE[0], RESIZE[1]))
        print(new.shape)
        if len(det):
            det[:, :4] = scale_coords(im.shape[2:], det[:, :4], img0.shape).round()
            # print(det)
            for j in det:
                x_start, y_start, x_end, y_end = (
                    transform(j[0], real_height, RESIZE[0]),
                    transform(j[1], real_width, RESIZE[1]),
                    transform(j[2], real_height, RESIZE[0]),
                    transform(j[3], real_width, RESIZE[1])
                )
                class_idx = int(j[-1])
                conf = j[-2]
                new = cv2.rectangle(new, (x_start, y_start), (x_end, y_end), COLOR_BORDER, 2)
                if y_start <50:
                    new = cv2.putText(new, real_classes[int(class_idx)]+" "+str(np.round(conf.numpy(),2)),(x_start, y_end+20),1,1.5,COLOR_TEXT,2)
                else : 
                   new = cv2.putText(new, real_classes[int(class_idx)]+" "+str(np.round(conf.numpy(),2)),(x_start, y_start-10),1,1.5,COLOR_TEXT,2)

    save_file_name = os.path.join(save_path,"detected.jpeg")
    cv2.imwrite(save_file_name,new)
    # print("Here")
    return render_template("next_page.html",img = "static/"+"detected.jpeg", real = "static/"+"raw.jpeg")


@app.route('/sucess')
def res():
    return render_template('next_page.html',result = "done") 

if __name__ == "__main__":
    WEIGHTS_PATH = (r"/home/vishwesh/Desktop/study/sem7/Project/content/yolov5/best.pt")
    DEVICE = "cpu"
    half = False
    model = DetectMultiBackend(WEIGHTS_PATH, device=DEVICE, dnn=False)
    app.run(debug=True)