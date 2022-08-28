from pyexpat.model import XML_CTYPE_CHOICE
from sys import flags
import cv2
import pathlib

cascade_path =pathlib.Path(cv2.__file__).parent.absolute() / "data_haarcascade_frontalface_default.xml"

clf= cv2.CascadeClassifier(str(cascade_path))

camera=cv2.VideoCapture(0)

while True :
  _, frame = camera.read()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  faces = clf.detectMultiScale(
    gray,
    scaleFactor = 1.1,
    minNeighbors=5,
    minSize=(30,30),
    flags=cv2.CASCADE_SCALE_IMAGE
    
    
  )