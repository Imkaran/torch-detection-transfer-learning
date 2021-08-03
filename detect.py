import cv2
import sys
import time
import torch
import argparse
import numpy as np
from torchvision.models import detection
from imutils.video import VideoStream, FPS

MODELS = {
    "frcnn-resnet":detection.fasterrcnn_resnet50_fpn,
    "frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn,
    "retinanet": detection.retinanet_resnet50_fpn
}

class ObjectDetection():
    def __init__(self):
        with open("coco_classes") as f:
            data = f.read()

        self.CLASSES = data.split("\n")
        self.NUM_CLASSES = len(self.CLASSES)
        self.COLORS = np.random.uniform(0, 255, size=(self.NUM_CLASSES, 3))
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.args = self.argumentParser()
        self.model = self.modelSelection()

        if self.args["stream"].lower().strip() == 'image':
            self.rawDetection()
        else:
            self.realTimeDetection()

    def argumentParser(self):
        ap = argparse.ArgumentParser()
        ap.add_argument("-i", "--image", type=str, default=None,
                        help="path to the input image")
        ap.add_argument("-m", "--model", type=str, default="frcnn-mobilenet",
                        choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
                        help="name of the object detection model")
        ap.add_argument("-c", "--confidence", type=float, default=0.5,
                        help="minimum probability to filter weak detections")
        ap.add_argument("-s", "--stream", type=str, default="image",
                        choices=["image", "webcam"],
                        help="raw image or webcam to be used for detection")
        args = vars(ap.parse_args())

        return args

    def modelSelection(self):
        model = MODELS[self.args["model"]](pretrained=True, progress=True, num_classes=self.NUM_CLASSES,
                                          pretrained_backbone=True)
        model = model.to(self.device)
        model.eval()

        return model

    def rawDetection(self):
        image = cv2.imread(self.args['image'])
        original = image.copy()
        image = self.preprocessImage(image)
        detection = self.model(image)[0]
        original = self.plotDetection(original, detection)

        cv2.imshow("Output", original)
        cv2.waitKey(0)

    def realTimeDetection(self):
        vs = VideoStream(src=0).start()
        time.sleep(2.0)
        fps = FPS().start()

        while True:
            frame = vs.read()
            original = frame.copy()
            frame = self.preprocessImage(frame)
            detection = self.model(frame)[0]
            original = self.plotDetection(original, detection)
            cv2.imshow("Camera", original)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            fps.update()

        fps.stop()
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        cv2.destroyAllWindows()
        vs.stop()

    def preprocessImage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.transpose((2, 0, 1))
        image = np.expand_dims(image, axis=0)
        image = image / 255.
        image = torch.FloatTensor(image)
        image = image.to(self.device)
        return image

    def plotDetection(self, image, detection):

        for i in range(len(detection['boxes'])):
            confidence = detection["scores"][i]

            if confidence > self.args['confidence']:
                idx = detection['labels'][i]
                bbox = detection['boxes'][i].detach().cpu().numpy()
                x1 = bbox[0].astype('int')
                y1 = bbox[1].astype('int')
                x2 = bbox[2].astype('int')
                y2 = bbox[3].astype('int')

                cv2.rectangle(image, (x1, y1), (x2, y2), self.COLORS[i], 2)
                label = "{}:{:.2f}%".format(self.CLASSES[idx], confidence * 100)
                y = y1 - 15 if y1 - 15 > 15 else y1 + 15
                cv2.putText(image, label, (x1, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLORS[idx], 2)
        return image


if __name__=='__main__':
    obj = ObjectDetection()
    sys.exit(0)
