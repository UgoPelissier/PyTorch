# import the necessary packages
from torchvision.models import detection
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import pickle
import torch
import time
import cv2
import os

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", type=str, required=True,
	help="path to the input video")
ap.add_argument("-m", "--model", type=str, default="frcnn-resnet",
	choices=["frcnn-resnet", "frcnn-mobilenet", "retinanet"],
	help="name of the object detection model")
ap.add_argument("-l", "--labels", type=str, default="coco_classes.pickle",
	help="path to file containing list of categories in COCO dataset")
ap.add_argument("-c", "--confidence", type=float, default=0.5,
	help="minimum probability to filter weak detections")
args = vars(ap.parse_args())

# set the video path
VIDEO_PATH = "/data/users/upelissier/10-Examples/Object_Detection/video"

# set the device we will be using to run the model
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the list of categories in the COCO dataset and then generate a
# set of bounding box colors for each class
CLASSES = pickle.loads(open(args["labels"], "rb").read())
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

# initialize a dictionary containing model name and its corresponding torchvision function call specify path to the loaded models
MODELS_PATH = "/data/users/upelissier/10-Examples/Object_Detection/models"

MODELS = {
	"frcnn-resnet": detection.fasterrcnn_resnet50_fpn(path=os.path.join(MODELS_PATH,"fasterrcnn_resnet50_fpn_coco-258fb6c6.pth"), progress=True, num_classes=len(CLASSES)),
	"frcnn-mobilenet": detection.fasterrcnn_mobilenet_v3_large_320_fpn(path=os.path.join(MODELS_PATH,"fasterrcnn_mobilenet_v3_large_320_fpn-907ea3f9.pth"), progress=True, num_classes=len(CLASSES)),
	"retinanet": detection.retinanet_resnet50_fpn(path=os.path.join(MODELS_PATH,"retinanet_resnet50_fpn_coco-eeacb38b.pth"), progress=True, num_classes=len(CLASSES))
}

# load the model and set it to evaluation mode
model = MODELS[args["model"]].to(DEVICE)
model.eval()

# initialize the video stream, allow the camera sensor to warmup, and initialize the FPS counter
print("[INFO] starting video stream...")
video = os.path.join(VIDEO_PATH,args["video"])
print("[INFO] video: {}".format(video))

vs = cv2.VideoCapture(video)
if (vs.isOpened() == False):
  print("Error opening the video file")
# Read fps and frame count
else:
  # Get frame rate information
  fps = vs.get(5)
  print('[INFO] frames per second : {} FPS'.format(fps))

time.sleep(2.0)
fps = FPS().start()

ret, frame = vs.read()
orig = frame.copy()

fshape = orig.shape
fheight = fshape[0]
fwidth = fshape[1]
print("[INFO] shape: [{},{}]".format(fheight,fwidth))

fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('output.avi',fourcc, 20.0, (fwidth,fheight))

# loop over the frames from the video stream
it = 0
while True:
	# grab the frame from the threaded video stream and resize it to have a maximum width of 400 pixels
	ret, frame = vs.read()
	if (frame is None):
		break
	orig = frame.copy()

	# convert the frame from BGR to RGB channel ordering and change
	# the frame from channels last to channels first ordering
	frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	frame = frame.transpose((2, 0, 1))

	# add a batch dimension, scale the raw pixel intensities to the
	# range [0, 1], and convert the frame to a floating point tensor
	frame = np.expand_dims(frame, axis=0)
	frame = frame / 255.0
	frame = torch.FloatTensor(frame)

	# send the input to the device and pass the it through the
	# network to get the detections and predictions
	frame = frame.to(DEVICE)
	detections = model(frame)[0]

	# loop over the detections
	for i in range(0, len(detections["boxes"])):
		# extract the confidence (i.e., probability) associated with
		# the prediction
		confidence = detections["scores"][i]

		# filter out weak detections by ensuring the confidence is
		# greater than the minimum confidence
		if confidence > args["confidence"]:
			# extract the index of the class label from the
			# detections, then compute the (x, y)-coordinates of
			# the bounding box for the object
			idx = int(detections["labels"][i])
			box = detections["boxes"][i].detach().cpu().numpy()
			(startX, startY, endX, endY) = box.astype("int")
            
			# draw the bounding box and label on the frame
			label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
			cv2.rectangle(orig, (startX, startY), (endX, endY), COLORS[idx], 2)
			y = startY - 15 if startY - 15 > 15 else startY + 15
			cv2.putText(orig, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

	# show the output frame
	out.write(orig)
	# cv2.imshow("Frame", orig)
	key = cv2.waitKey(1) & 0xFF

	# if the 'q' key was pressed, break from the loop
	if key == ord("q"):
		break

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
out.release()
cv2.destroyAllWindows()