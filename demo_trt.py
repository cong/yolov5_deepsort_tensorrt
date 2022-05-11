# -*- coding: utf-8 -*-
#!/usr/bin/python3
"""
Created on 2021/5/24 13:46
@Author: Wang Cong
@Email : iwangcong@outlook.com
@Version : 0.1
@File : demo_trt.py
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
import time
import ctypes
import tracker_trt
from detector_trt import Detector


def detect(video_path, engine_file_path):
    detector = Detector(engine_file_path)
    # capture = cv2.VideoCapture(video_path)
    capture = cv2.VideoCapture(0)
    fps = 0.0
    while True:
        ret, img = capture.read()
        if img is None:
            print('No image input!')
            break

        t1 = time.time()
        bboxes = detector.detect(img)
        # print((time.time() - t1) * 1000)

        if len(bboxes) > 0:
            # t2 = time.time()
            list_bboxs = tracker_trt.update(bboxes, img)
            # print((time.time() - t2) * 1000)
            output_image_frame = tracker_trt.draw_bboxes(img, list_bboxs, line_thickness=None)
        else:
            output_image_frame = img

        fps = (fps + (1. / (time.time() - t1))) / 2
        cv2.putText(output_image_frame, 'FPS: {:.2f}'.format(fps), (50, 30), 0, 1, (0, 255, 0), 2)
        cv2.putText(output_image_frame, 'Time: {:.3f}'.format(time.time() - t1), (50, 60), 0, 1, (0, 255, 0), 2)
        if ret == True:
            cv2.imshow('frame', output_image_frame)
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        else:
            break

    capture.release()
    cv2.destroyAllWindows()
    detector.destroy()


if __name__ == '__main__':

    video_path = './video/test.mp4'
    PLUGIN_LIBRARY = "./weights/libmyplugins.so"
    ctypes.CDLL(PLUGIN_LIBRARY)
    engine_file_path = './weights/yolov5s.engine'
    detect(video_path, engine_file_path)

