from ultralytics import YOLO
import pandas as pd
import os


class ObjectDetectionHelper:

    def detectionTrainedClasses(self,yolo_model,images_source,conf=0.5):

        model = YOLO(yolo_model)
        entries = os.listdir(images_source)
        for entry in entries:
            file_path = images_source + '/' + entry
            results = model.predict(source=file_path, conf= conf, save=True, save_conf=True,
                                    save_txt=True, exist_ok=True)  # save predictions as detected_objects


