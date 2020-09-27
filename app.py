# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from flask_environments import Environments
import os
from keras_retinanet import models

# Define a flask app
class DevelopmentConfig(object):
    ENV = 'development'
    DEBUG = True

MODEL_PATH = 'C:/Users/karim/Desktop/License-Plate-Detection-Recognition/snapshots/resnet50_full.h5'

def load_inference_model(model_path=MODEL_PATH):
        #print(model_path)
        model = models.load_model(model_path, backbone_name='resnet50')
        model = models.convert_model(model)
        #model.summary()
        return model
model = load_inference_model(MODEL_PATH)
#DATABASE
import pymongo
from pymongo import MongoClient

cluster=MongoClient("mongodb+srv://karim:Karim123@cluster0-ptg1l.mongodb.net/test?retryWrites=true&w=majority")
db= cluster["data"]
collection=db["data"]



app = Flask(__name__)
app.config.from_object(DevelopmentConfig)

def func(impath):
    import cv2 as cv

    import sys
    import numpy as np
    import os.path

 

    # Initializing the parameters
    confThreshold = 0.5  #Confidence threshold
    nmsThreshold = 0.4  #Non-maximum suppression threshold

    inpWidth = 416       #Width of network's input image
    inpHeight = 416      #Height of network's input image


    # Give the configuration and weight files for the model and load the network using them.

    modelconfg = "C:/Users/karim/Desktop/License-Plate-Detection-Recognition/Licence_plate_detection/darknet-yolov3.cfg"
    Weights = "C:/Users/karim/Desktop/License-Plate-Detection-Recognition/Licence_plate_detection//lapi.weights"

    net = cv.dnn.readNetFromDarknet(modelconfg, Weights)
    net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)



    # Get the names of the output layers
    def getOutputlayersNames(net):
        # Geting the names of all the layers in the network
        layersNames = net.getLayerNames()
        # Geting the names of the output layers
        return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    # Draw the predicted bounding box
    def drawBox( frame,left, top, right, bottom):
        # Draw a bounding box.
        cv.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 3)

    # Removing bounding boxes with low confidence using non-maxima suppression
    def postprocess(frame, outputs):
        frameHeight = frame.shape[0]
        frameWidth = frame.shape[1]
        # Scaning through all the bounding boxes output from the network and keeping only the
        # ones with high confidence scores.
        classesIds = []
        confidences = []
        Bboxes = []
        for out in outputs:
            #print("out.shape : ", out.shape)
            for detection in out:
                #if detection[4]>0.001:
                scores = detection[5:]
                classId = np.argmax(scores)
                #if scores[classId]>confThreshold:
                confidence = scores[classId]
                if detection[4]>confThreshold:
                    #print(detection[4], " - ", scores[classId], " - th : ", confThreshold)
                    print(detection)
                if confidence > confThreshold:
                    x = int(detection[0] * frameWidth)
                    y = int(detection[1] * frameHeight)
                    width = int(detection[2] * frameWidth)
                    height = int(detection[3] * frameHeight)
                    left = int(x - width / 2)
                    top = int(y - height / 2)
                    classesIds.append(classId)
                    confidences.append(float(confidence))
                    Bboxes.append([left, top, width, height])
                    
        # Perform non maximum suppression to eliminate redundant overlapping boxes with
        # lower confidences.
        indices = cv.dnn.NMSBoxes(Bboxes, confidences, confThreshold, nmsThreshold)
        for i in indices:
            i = i[0]
            box = Bboxes[i]
            left = box[0]
            top = box[1]
            width = box[2]
            height = box[3]
            drawBox( frame,left, top, left + width, top + height)
            coordinates=[left,top, left + width, top + height]
            return coordinates
    
    frame=cv.imread(impath)


    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1/255, (inpWidth, inpHeight), [0,0,0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outputs = net.forward(getOutputlayersNames(net))

    # Remove the bounding boxes with low confidence
    coordinates=postprocess(frame, outputs)


    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)

    #cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))

    # Write the frame with the detection boxes


    # -*- coding: utf-8 -*-

    from keras_retinanet import models
    from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image

    import matplotlib.pyplot as plt
    import cv2
    import os
    import numpy as np
    import time

    #Digits detection and recognition Part
    def post_processing(boxes, original_img, preprocessed_img):
        #post-processing
        h, w, _ = preprocessed_img.shape
        h2, w2, _ = original_img.shape
        boxes[:, :, 0] = boxes[:, :, 0] / w * w2
        boxes[:, :, 2] = boxes[:, :, 2] / w * w2
        boxes[:, :, 1] = boxes[:, :, 1] / h * h2
        boxes[:, :, 3] = boxes[:, :, 3] / h * h2
        return boxes


    #A set of functions that are used for visualization.

    #These functions often receive an image, perform some visualization on the image.
    #The functions do not return a value, instead they modify the image itself.

    
    import collections
    import numpy as np
    import PIL.Image as Image
    import PIL.ImageColor as ImageColor
    import PIL.ImageDraw as ImageDraw
    import PIL.ImageFont as ImageFont


    _TITLE_LEFT_MARGIN = 10
    _TITLE_TOP_MARGIN = 10
    STANDARD_COLORS = [
            'AliceBlue', 'Chartreuse', 'Aqua', 'Aquamarine', 'Azure', 'Beige', 'Bisque',
            'BlanchedAlmond', 'BlueViolet', 'BurlyWood', 'CadetBlue', 'AntiqueWhite',
            'Chocolate', 'Coral', 'CornflowerBlue', 'Cornsilk', 'Crimson', 'Cyan',
            'DarkCyan', 'DarkGoldenRod', 'DarkGrey', 'DarkKhaki', 'DarkOrange',
            'DarkOrchid', 'DarkSalmon', 'DarkSeaGreen', 'DarkTurquoise', 'DarkViolet',
            'DeepPink', 'DeepSkyBlue', 'DodgerBlue', 'FireBrick', 'FloralWhite',
            'ForestGreen', 'Fuchsia', 'Gainsboro', 'GhostWhite', 'Gold', 'GoldenRod',
            'Salmon', 'Tan', 'HoneyDew', 'HotPink', 'IndianRed', 'Ivory', 'Khaki',
            'Lavender', 'LavenderBlush', 'LawnGreen', 'LemonChiffon', 'LightBlue',
            'LightCoral', 'LightCyan', 'LightGoldenRodYellow', 'LightGray', 'LightGrey',
            'LightGreen', 'LightPink', 'LightSalmon', 'LightSeaGreen', 'LightSkyBlue',
            'LightSlateGray', 'LightSlateGrey', 'LightSteelBlue', 'LightYellow', 'Lime',
            'LimeGreen', 'Linen', 'Magenta', 'MediumAquaMarine', 'MediumOrchid',
            'MediumPurple', 'MediumSeaGreen', 'MediumSlateBlue', 'MediumSpringGreen',
            'MediumTurquoise', 'MediumVioletRed', 'MintCream', 'MistyRose', 'Moccasin',
            'NavajoWhite', 'OldLace', 'Olive', 'OliveDrab', 'Orange', 'OrangeRed',
            'Orchid', 'PaleGoldenRod', 'PaleGreen', 'PaleTurquoise', 'PaleVioletRed',
            'PapayaWhip', 'PeachPuff', 'Peru', 'Pink', 'Plum', 'PowderBlue', 'Purple',
            'Red', 'RosyBrown', 'RoyalBlue', 'SaddleBrown', 'Green', 'SandyBrown',
            'SeaGreen', 'SeaShell', 'Sienna', 'Silver', 'SkyBlue', 'SlateBlue',
            'SlateGray', 'SlateGrey', 'Snow', 'SpringGreen', 'SteelBlue', 'GreenYellow',
            'Teal', 'Thistle', 'Tomato', 'Turquoise', 'Violet', 'Wheat', 'White',
            'WhiteSmoke', 'Yellow', 'YellowGreen'
    ]


    def draw_boxes_on_image_array(image, 
                                         ymin,
                                         xmin,
                                         ymax,
                                         xmax,
                                         color='red',
                                         thickness=4,
                                         display_str_list=(),
                                         use_normalized_coordinates=True):

        image_pil = Image.fromarray(np.uint8(image)).convert('RGB')
        
        draw_boxes_on_image(image_pil, ymin, xmin, ymax, xmax, color,
                            thickness, display_str_list,
                            use_normalized_coordinates)
        np.copyto(image, np.array(image_pil))


    def draw_boxes_on_image(image,
                            ymin,
                            xmin,
                            ymax,
                            xmax,
                            color='red',
                            thickness=4,
                            display_str_list=(),
                            use_normalized_coordinates=True):
        draw = ImageDraw.Draw(image)
        im_width, im_height = image.size
        if use_normalized_coordinates:
            (left, right, top, bottom) = (xmin * im_width, xmax * im_width,
                                                                        ymin * im_height, ymax * im_height)
        else:
            (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
        draw.line([(left, top), (left, bottom), (right, bottom),
                            (right, top), (left, top)], width=thickness, fill=color)
        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except IOError:
            font = ImageFont.load_default()

        # If the total height of the display strings added to the top of the bounding
        # box exceeds the top of the image, stack the strings below the bounding box
        # instead of above.
        display_str_heights = [font.getsize(ds)[1] for ds in display_str_list]
        # Each display_str has a top and bottom margin of 0.05x.
        total_display_str_height = (1 + 2 * 0.05) * sum(display_str_heights)

        if top > total_display_str_height:
            text_bottom = top
        else:
            text_bottom = bottom + total_display_str_height
        # Reverse list and print from bottom to top.
        
        for display_str in display_str_list[::-1]:
            text_width, text_height = font.getsize(display_str)
            margin = np.ceil(0.05 * text_height)
            draw.rectangle(
                    [(left, text_bottom - text_height - 2 * margin), (left + text_width,
                                                                                                                        text_bottom)],
                    fill=color)
            draw.text(
                    (left + margin, text_bottom - text_height - margin),
                    display_str,
                    fill='black',
                    font=font)
            
            
            text_bottom -= text_height - 2 * margin
        
    def draw_mask_on_image_array(image, mask, color='red', alpha=0.4):
        """Draws mask on an image.
        Args:
            image: uint8 numpy array with shape (img_height, img_height, 3)
            mask: a uint8 numpy array of shape (img_height, img_height) with
                values between either 0 or 1.
            color: color to draw the keypoints with. Default is red.
            alpha: transparency value between 0 and 1. (default: 0.4)
        Raises:
            ValueError: On incorrect data type for image or masks.
        """
        if image.dtype != np.uint8:
            raise ValueError('`image` not of type np.uint8')
        if mask.dtype != np.uint8:
            raise ValueError('`mask` not of type np.uint8')
        if np.any(np.logical_and(mask != 1, mask != 0)):
            raise ValueError('`mask` elements should be in [0, 1]')
        if image.shape[:2] != mask.shape:
            raise ValueError('The image has spatial dimensions %s but the mask has '
                                            'dimensions %s' % (image.shape[:2], mask.shape))
        rgb = ImageColor.getrgb(color)
        pil_image = Image.fromarray(image)

        solid_color = np.expand_dims(np.ones_like(mask), axis=2) * np.reshape(list(rgb), [1, 1, 3])
        pil_solid_color = Image.fromarray(np.uint8(solid_color)).convert('RGBA')
        pil_mask = Image.fromarray(np.uint8(255.0*alpha*mask)).convert('L')
        pil_image = Image.composite(pil_solid_color, pil_image, pil_mask)
        np.copyto(image, np.array(pil_image.convert('RGB')))


    def visualize_boxes_and_labels_on_image(
            image,
            boxes,
            classes,
            scores,
            category_index,
            instance_masks=None,
            instance_boundaries=None,
            use_normalized_coordinates=False,
            max_boxes_to_draw=20,
            min_score_thresh=.5,
            agnostic_mode=False,
            line_thickness=4,
            groundtruth_box_visualization_color='black',
            skip_scores=False,
            skip_labels=False):
        # Create a display string (and color) for every box location, group any boxes
        # that correspond to the same location.
        box_to_display_str_map = collections.defaultdict(list)
        box_to_color_map = collections.defaultdict(str)
        box_to_instance_masks_map = {}
        box_to_instance_boundaries_map = {}
        if not max_boxes_to_draw:
            max_boxes_to_draw = boxes.shape[0]
        for i in range(min(max_boxes_to_draw, boxes.shape[0])):
            if scores is None or scores[i] > min_score_thresh:
                box = tuple(boxes[i].tolist())
                if instance_masks is not None:
                    box_to_instance_masks_map[box] = instance_masks[i]
                if instance_boundaries is not None:
                    box_to_instance_boundaries_map[box] = instance_boundaries[i]
                if scores is None:
                    box_to_color_map[box] = groundtruth_box_visualization_color
                else:
                    display_str = ''
                    if not skip_labels:
                        if not agnostic_mode:
                            if classes[i] in category_index.keys():
                                class_name = category_index[classes[i]]['name']
                            else:
                                class_name = 'N/A'
                            display_str = str(class_name)
                    if not skip_scores:
                        if not display_str:
                            display_str = '{}%'.format(int(100*scores[i]))
                        else:
                            display_str = '{}: {}%'.format(display_str, int(100*scores[i]))
                    box_to_display_str_map[box].append(display_str)
                    if agnostic_mode:
                        box_to_color_map[box] = 'DarkOrange'
                    else:
                        box_to_color_map[box] = STANDARD_COLORS[
                                classes[i] % len(STANDARD_COLORS)]

        # Draw all boxes
        result=[]
        for box, color in box_to_color_map.items():
            ymin, xmin, ymax, xmax = box
            if instance_masks is not None:
                draw_mask_on_image_array(
                        image,
                        box_to_instance_masks_map[box],
                        color=color
                )
            if instance_boundaries is not None:
                draw_mask_on_image_array(
                        image,
                        box_to_instance_boundaries_map[box],
                        color='red',
                        alpha=1.0
                )
            result.append([
                    xmin,
                    box_to_display_str_map[box][0][0]])
                    
            draw_boxes_on_image_array(
                    image,
                    ymin,
                    xmin,
                    ymax,
                    xmax,
                    color=color,
                    thickness=line_thickness,
                    display_str_list=box_to_display_str_map[box],
                    use_normalized_coordinates=use_normalized_coordinates)
        def sorting(result):
            v=[]
            res=[]
            minx=[]
            dif=[]
            for i in range(0, len (result)):
                v.append(result[i][0])
            v.sort()
            for i in v:
                for j in result:
                    if i == j[0]:
                        res.append(j[1])
                        minx.append(j[0])
            for i in range(0,len(minx)-1):
                dif.append(minx[i+1]-minx[i])
            p=np.argmax(dif)+1
            res.insert(p,'T') 
            return res
        
        pred=sorting(result)
        return pred

    def visualize_boxes(image, boxes, labels, probs, class_labels):
        category_index = {}
        
        for id_, label_name in enumerate(class_labels):
            category_index[id_] = {"name": label_name}
        
        boxes = np.array([np.array([b[1],b[0],b[3],b[2]]) for b in boxes])
        
        pred=visualize_boxes_and_labels_on_image(image, boxes, labels, probs, category_index)
        return pred



    if __name__ == '__main__':
        
            

            image=frame[coordinates[1]:coordinates[3],coordinates[0]:coordinates[2]]
            draw = image.copy()
            draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

            # preprocess image for network
            image = preprocess_image(image)
            image, _ = resize_image(image, 416, 448)

            # process image
            start = time.time()
            boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
            print("processing time: ", time.time() - start)

            boxes = post_processing(boxes, draw, image)
        
            labels = labels[0] 
            scores = scores[0]
            boxes = boxes[0]
            
            
            #pred=[]
            pred=visualize_boxes(draw, boxes, labels, scores, class_labels=['0', '1', '2', '3', '4', '5', '6', '7', '8', '9'])
            res=''
            for ele in pred:  
                res += ele 
    return res
@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        impath = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(impath)

        # Make prediction
        res = func(impath)
        rest=collection.find({"serie":res})
        ch=''
        for i in rest:
            ch=i['cin']
        cat=res+'=> Person CIN:'+ch
        print(cat)
        return cat
    return None


if __name__ == '__main__':
    app.run(debug=False, threaded=False)




