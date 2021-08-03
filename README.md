# torch-detection-transfer-learning
Object detection using PyTorch Transfer Learning

### Python Version:
Python 3.8

### Run detection with raw image:
```python object_detection.py -i sample.jpg```

### Run detection with webcam:
```python detect.py --stream webcam```

### optional arguments:
```
  -h, --help            show this help message and exit
  -i IMAGE, --image IMAGE
                        path to the input image
  -m {frcnn-resnet,frcnn-mobilenet,retinanet}, --model {frcnn-resnet,frcnn-mobilenet,retinanet}
                        name of the object detection model
  -c CONFIDENCE, --confidence CONFIDENCE
                        minimum probability to filter weak detections
  -s {image, webcam}, --stream {image,webcam}
                        raw image or webcam to be used for detection
```                        
