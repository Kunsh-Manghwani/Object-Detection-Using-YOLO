import cv2
import argparse
from utils import process_frame
CONF_THRESHOLD = 0.3
NMS_THRESHOLD = 0.4

parser = argparse.ArgumentParser()
parser.add_argument('--image', help='Path to input image', required=True)
args = parser.parse_args()
image = cv2.imread(args.image)
blob = cv2.dnn.blobFromImage(image,1 / 255.0, (416, 416), swapRB=True, crop=False)
with open('coco.names', 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg", "darknet")
outNames = net.getUnconnectedOutLayersNames()
net.setInput(blob)


outs = net.forward(outNames)
#print(outs[0])
process_frame(image, outs, classes, CONF_THRESHOLD, NMS_THRESHOLD)

cv2.imshow("Output", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("out.png", image)