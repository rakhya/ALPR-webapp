import numpy as np
import tensorflow as tf
from PIL import Image
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from object_detection.utils import ops as utils_ops
from object_detection.utils import label_map_util
import cv2
from collections import OrderedDict
from collections import defaultdict
from io import StringIO
import os
import six.moves.urllib as urllib
import sys
import tarfile
import zipfile
import pytesseract
import time
import datetime
"""from random import randint,choice"""

config = tf.ConfigProto()
config.gpu_options.allow_growth = True

app = Flask(__name__)



GAUSSIAN_SMOOTH_FILTER_SIZE = (5, 5)
ADAPTIVE_THRESH_BLOCK_SIZE = 11
ADAPTIVE_THRESH_WEIGHT = 2

MODEL_NAME = 'license_plate_inference_graph'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = MODEL_NAME +'/object_detection.pbtxt'

NUM_CLASSES = 1

char_dict = {}
chars = list(range(0,10))+list(map(chr,range(65,79)))+list(map(chr,range(80,91)))
for i,char in enumerate(chars):
    char_dict[i]=char

test_images_path = "test_images"
	
detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
category_index = label_map_util.create_category_index(categories)


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)

from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

class CNN:
    def __init__(self):
        self.model = None
        
    def CNN_initialize(self):
        # Initialising the CNN
        classifier = Sequential()

        # Step 1 - Convolution
        classifier.add(Convolution2D(32, 3, 3, input_shape = (20,20, 3), activation = 'relu'))

        # Step 2 - Pooling
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Adding a second convolutional and pooling layer
        classifier.add(Convolution2D(32, 3, 3, input_shape=(20,20,3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))

        # Step 3 - Flattening
        classifier.add(Flatten())

        # Step 4 - Full connection
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = 35, activation = 'softmax'))

        fname = "weights-Test-CNN6.hdf5"
        classifier.load_weights(fname)

        self.model = classifier

    def getModel(self):
        return self.model

def run_inference_for_single_image(image, graph):
    with graph.as_default():
        with tf.Session(config=config) as sess:
      # Get handles to input and output tensors
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
            for key in ['num_detections', 'detection_boxes', 'detection_scores','detection_classes', 'detection_masks']:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
                    tensor_name)
            if 'detection_masks' in tensor_dict:
        # The following processing is only for single image
                detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
                detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
        # Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
                real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
                detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
                detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
                detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
                detection_masks, detection_boxes, image.shape[0], image.shape[1])
                detection_masks_reframed = tf.cast(
                tf.greater(detection_masks_reframed, 0.5), tf.uint8)
        # # Follow the convention by adding back the batch dimension
                tensor_dict['detection_masks'] = tf.expand_dims(
                detection_masks_reframed, 0)
            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')

      # # Run inference
            output_dict = sess.run(tensor_dict,feed_dict={image_tensor: np.expand_dims(image, 0)})

      # all outputs are float32 numpy arrays, so convert types as appropriate
            output_dict['num_detections'] = int(output_dict['num_detections'][0])
            output_dict['detection_classes'] = output_dict['detection_classes'][0].astype(np.uint8)
            output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
            output_dict['detection_scores'] = output_dict['detection_scores'][0]
            if 'detection_masks' in output_dict:
                output_dict['detection_masks'] = output_dict['detection_masks'][0]
                
    return(output_dict)
	
def tesseract_predict(img):
    text = pytesseract.image_to_string(img,lang='eng')
    text = ''.join(e for e in text if e.isalnum())
    return text.upper()

def cnn_char_predict(imag,model):
    imag = cv2.resize(imag,(20,20))
    imag = imag.reshape((1,20,20,3))
    Y_pred = model.predict(imag)
    y_pred = np.argmax(Y_pred, axis=1)
    return str(char_dict[y_pred[0]])
    
def image_processing(im,plate_num,model):

    
    tess = {}
    #gray, thresh1 = preprocess(im)
    #im = cv2.imread(im)
    imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    resized_img = cv2.resize(imgray
                             , None
                             , fx=5.0
                             , fy=5.0
                             , interpolation=cv2.INTER_CUBIC)

    ret, thresh2 = cv2.threshold(resized_img, 127, 255, cv2.THRESH_BINARY)
    inv = cv2.bitwise_not(thresh2)
    height, weight = thresh2.shape
    area = weight * height
    char_contours = []
    char_mask = np.zeros_like(thresh2)
    char_mask2 = np.zeros_like(thresh2)
    validContours = []
    validContours1 = []
    validContours2 = []
    validContours3 = []
    contours, hierarcy = findAllContoursExternal(inv)
    for contour in contours:
        if(isValidContour(contour,area)):
            validContours1.append(contour)

    contours, hierarcy = findAllContoursTree(inv)
    for contour in contours:
        if (isValidContour(contour, area)):
            validContours2.append(contour)



    max_len = 0
    validContours = validContours1
    if len(validContours1) < 7:
        clean_img = clean_image(resized_img)
        bw_image = cv2.bitwise_not(clean_img)
        contours, hierarcy = findAllContoursTree(bw_image)
        for contour in contours:
            if (isValidContour(contour, area)):
                validContours3.append(contour)
        for i,cons in enumerate([validContours2,validContours3]):
            if len(cons) > max_len:
                max_len = len(cons)
                validContours = cons
                if (i == 1):
                    inv = bw_image

    coord = {}
    center = height/2
    for validContour in validContours:
        rect = cv2.minAreaRect(validContour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        x, y, w, h = cv2.boundingRect(validContour)
        centerX = x+w/2
        coord[centerX] = [x,y,w,h]
        roi = inv[y:y+h,x:x+w]
        #print(y,h)
        y1 = y
        y2 = y+h
        cont_center = (y1+y2)/2
        #print(center,cont_center)
        y = int(y + center - cont_center)
        char_mask2[y:y+roi.shape[0], x:x + roi.shape[1]] = roi
        #char_contours.append(validContour)
        cv2.drawContours(char_mask, [box], -1, 255, -1)
        #cv2.rectangle(imgray, (x, y), (x + w, y + h), 0, 1)

    coord = OrderedDict(sorted(coord.items()))

    clean = cv2.bitwise_not(cv2.bitwise_and(char_mask, char_mask, mask=inv))
    clean2 = cv2.bitwise_not(char_mask2)
    
    cnn_plate_number = ""
    for i,(k,v) in enumerate(coord.items()):
        x,y,w,h = v
        imag = clean[y:y+h,x:x+w]
        imag =cv2.cvtColor(imag,cv2.COLOR_GRAY2RGB)
        cnn_plate_number += cnn_char_predict(imag,model)
    
    tesseract_plate_number = tesseract_predict(clean2)
    
    #print(plate_num+": "+cnn_plate_number+"   "+tesseract_plate_number)
	
    plate = tesseract_plate_number
    if len(tesseract_plate_number) < len(cnn_plate_number):
        plate = cnn_plate_number
	
    cv2.destroyAllWindows()
    return plate
    
def reduce_colors(img, n):
    Z = img.reshape((-1,3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = n
    ret,label,center=cv2.kmeans(Z,K,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img.shape))

    return res2

    # ============================================================================

def clean_image(imgGrayResized):

    resized_img = cv2.GaussianBlur(imgGrayResized,(5,5),0)
    #cv2.imwrite(PLATE_PROCESS_PATH+'/licence_plate_large_{}.png'.format(i), resized_img)

    equalized_img = cv2.equalizeHist(resized_img)
    #cv2.imwrite(PLATE_PROCESS_PATH+'/licence_plate_equ_{}.png'.format(i), equalized_img)


    reduced = cv2.cvtColor(reduce_colors(cv2.cvtColor(equalized_img, cv2.COLOR_GRAY2BGR), 8), cv2.COLOR_BGR2GRAY)
    #reduced = cv2.cvtColor(resized_img, cv2.COLOR_BGR2GRAY)
    #imgBlurred = cv2.GaussianBlur(reduced, (5, 5), 0)
    #cv2.imwrite(PLATE_PROCESS_PATH+'/licence_plate_red_{}.png'.format(i), reduced)


    ret, mask = cv2.threshold(reduced, 64, 255, cv2.THRESH_BINARY)
    #cv2.imwrite(PLATE_PROCESS_PATH+'/licence_plate_mask_{}.png'.format(i), mask)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    mask = cv2.erode(mask, kernel, iterations = 1)
    #cv2.imwrite(PLATE_PROCESS_PATH+'/licence_plate_mask2_{}.png'.format(i), mask)

    return mask

def findAllContoursExternal(threshImg):
    _, contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours,hierarchy

def findAllContoursTree(threshImg):
    _, contours, hierarchy = cv2.findContours(threshImg, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    return contours,hierarchy

def isValidContour(contour,img_area):
    x, y, w, h = cv2.boundingRect(contour)
    cont_area = w * h
    if cont_area > img_area / 43 and cont_area < img_area / 5 and h > 1.2 * w and h < 5*w:
        return True
    return False

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        start_time = time.time()
        files = request.files.getlist("file[]")
        res = []
        cnn = CNN()
        cnn.CNN_initialize()
        model = cnn.getModel()
        for k,image_path in enumerate(files):
            image = Image.open(image_path)
            filename = secure_filename(image_path.filename)
            path = os.path.join(test_images_path, filename)
            image = image.convert("RGB")
  # the array based representation of the image will be used later in order to prepare the
  # result image with boxes and labels on it.
            image_np = load_image_into_numpy_array(image)
  # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
            image_np_expanded = np.expand_dims(image_np, axis=0)
  # Actual detection.
            output_dict = run_inference_for_single_image(image_np, detection_graph)
            w,h = image.size
            boxes = np.array(np.where(output_dict['detection_scores']>0.99))[0]
            plate_list = []
            for i,b in enumerate(boxes):
                ymin,xmin,ymax,xmax = output_dict['detection_boxes'][b]
                img = image.crop((xmin*w,ymin*h,xmax*w,ymax*h))
                img = np.array(img)
                plate_num = image_processing(img,"plate{}".format(i+1),model)
                plate_list.append(plate_num)
            for i in range(len(plate_list)):
                if i==0:
                    re = {"FileName":"<a href="+path+" target=\"_blank\">"+filename+"</a>","PlateName":"Plate1","LicensePlateNumber":plate_list[i]}
                else:
                    re = {"FileName":" ","PlateName":"Plate"+str(i+1),"LicensePlateNumber":plate_list[i]}
                res.append(re)
        columns = [{"field": "FileName", "title": "ImageName","sortable": True},{"field": "PlateName", "title": "Plate Name","sortable": True},{"field": "LicensePlateNumber", "title": "License Plate Number","sortable": True}]
        end_time = time.time()
        print("Processing time "+ str(datetime.timedelta(seconds=end_time-start_time)))
        return render_template("table.html",data=res,columns=columns, title='Prediction Results')

	
@app.route('/uploads')
def uploads():
	return render_template("upload.html")
	

@app.route('/')
def index():
	return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
