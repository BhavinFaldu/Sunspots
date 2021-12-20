from imutils import contours
from skimage import measure
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import argparse
import imutils
import cv2 
from PIL import Image, ImageOps
from sklearn.neighbors import NearestNeighbors
# construct the argument parse and parse the arguments
#from google.colab.patches import cv2_imshow
#from matplotlib import pyplot as plt
import math
import streamlit as st

st.title("Space Weather Prediction By Analysing Sunspots")
file = st.file_uploader("Please upload an Sun's image", type=["jpg", "png"])


if file is None:
    st.text("Please upload an image file")
else:
    
    with open(file.name,'wb') as f:
        f.write(file.read())
    st.write("Input image:")
    
    
    img = cv2.imread(file.name)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    st.image(img)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (11, 11), 0)
    st.write("Gray image:")
    st.image(gray)
    
    #thresold to detect bright region
    thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)[1]

    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=4)
    #cv2_imshow(thresh)

    # perform a connected component analysis on the thresholded
    # image, then initialize a mask to store only the "large"
    # components
    labels = measure.label(thresh, connectivity=2, background=0)
    #labels = measure.label(thresh, neighbors=8, background=0)
    mask = np.zeros(thresh.shape, dtype="uint8")

    # loop over the unique components
    for label in np.unique(labels):
      if label == 0:
		        continue

      labelMask = np.zeros(thresh.shape, dtype="uint8")
      labelMask[labels == label] = 255
      numPixels = cv2.countNonZero(labelMask)

      if numPixels > 300:
        mask = cv2.add(mask, labelMask)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = contours.sort_contours(cnts)[0]

    for (i, c) in enumerate(cnts):
      (x, y, w, h) = cv2.boundingRect(c)
      ((cX, cY), radius) = cv2.minEnclosingCircle(c)
      cv2.circle(img, (int(cX), int(cY)), int(radius),(0, 255, 0), 3)
      cv2.putText(img, "#{}".format(i + 1), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

    st.write("Output Image with Detected Sunspots:")
    st.image(img)
    st.write(f'we are able to detect {len(cnts)} Sunspots in this image.')