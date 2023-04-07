# PyTorch

## Basic
```
conda env create -f environment.yml
python train.py
```

## CNN
```
conda env create -f environment.yml
python train.py --model output/model.pth --plot output/plot.png
python predict.py --model output/model.pth
```

## Object_Detection
```
conda env create -f environment.yml
python classify_image.py --image images/soccer_ball.jpg --model inception
python detect_image.py --model frcnn-resnet --image images/example_01.jpg --labels coco_classes.pickle   
```

## Pre_Trained_Networks
```
conda env create -f environment.yml
python detect_realtime.py --video road_traffic_trim.mp4 --model frcnn-mobilenet --labels coco_classes.pickle
```
