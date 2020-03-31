import numpy as np
import dlib
import cv2
import argparse
import os
from image_utility import save_image, generate_random_color, draw_border
from imutils import face_utils

import matplotlib.pyplot as plt

WEIGHTS = './shape_predictor_68_face_landmarks.dat'
MODEL = 'hog'

face_detector = dlib.get_frontal_face_detector()
model = MODEL
predictor = dlib.shape_predictor(WEIGHTS)

def hog_landmarks(image, gray):
    faces_hog = face_detector(gray, 1)

    # HOG + SVN
    facial_points = []
    for (i, face) in enumerate(faces_hog):
        # Finding points for rectangle to draw on face
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Drawing simple rectangle around found faces
        #cv2.rectangle(image, (x, y), (x + w, y + h), generate_random_color(), 2)

        # Make the prediction and transfom it to numpy array
        shape = predictor(gray, face)
        shape = face_utils.shape_to_np(shape)

        # Draw on our image, all the finded cordinate points (x,y)
#         for (x, y) in shape:
#             cv2.circle(image, (x, y), 2, (0, 255, 0), -1)
            
        facial_points += [shape]
    
    return facial_points
            
def face_detection(image):

    # Converting the image to gray scale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # The 1 in the second argument indicates that we should upsample the image
    # 1 time. This will make everything bigger and allow us to detect more
    # faces.

    # write at the top left corner of the image
    img_height, img_width = image.shape[:2]
    if model == 'hog':
        facial_points = hog_landmarks(image, gray)
    elif model == 'cnn':
        cnn_landmarks(image, gray)
    else:
        dl_landmarks(image, gray, img_height, img_width)

    #save_image(image)
    return facial_points

def build_mask(lm):
    left_temple = 0.5*lm[0] + 0.5*lm[17]
    right_temple = 0.5*lm[26] + 0.5*lm[16]
    
    #left_jaw = lm[np.arange(0,8+1),:]
    left_jaw = np.vstack([
        lm[0],
        0.8*lm[0] + 0.2*lm[1],
        0.8*left_temple + 0.2*lm[4],
        0.2*left_temple + 0.8*lm[4],
        0.2*lm[2] + 0.8*lm[3],
        lm[3],
        0.2*left_temple + 0.8*lm[4],
        lm[5:9,:]
    ])
    
    
    #right_jaw = lm[np.arange(9,17), :]
    right_jaw = np.flipud(np.vstack([
        lm[16],
        0.5*lm[16] + 0.5*lm[15],
        0.7*right_temple + 0.3*lm[12],
        0.3*right_temple + 0.7*lm[12],
        0.5*lm[14] + 0.5*lm[13],
        lm[13],
        0.2*right_temple + 0.8*lm[12],
        lm[[11,10, 9, 8],:]
    ]))

    right_cheekbone = [0.6*lm[25] + 0.4*lm[11],
                        0.4*lm[23] + 0.6*lm[53],
                       0.3*lm[22] + 0.7*lm[35]]

    left_cheekbone = [0.3*lm[21] + 0.7*lm[31],
                        0.4*lm[20] + 0.6*lm[49],
                      0.6*lm[18] + 0.4*lm[4]]
    nose = lm[[29, 30]]
    lips = lm[[51,62, 66,57]]
    chin = lm[8]


    right_mask = np.vstack([
        right_jaw,
        right_cheekbone,
        nose,
        lips,
        chin
    ])

    left_mask = np.vstack([
        chin,
        np.flipud(lips),
        np.flipud(nose),
        left_cheekbone,
        left_jaw
    ])
    
    return left_mask, right_mask

def impose_mask(imfile, output_file):
    assert output_file[-4:] == '.jpg' or output_file[-5:] == '.jpeg'
    
    image = cv2.imread(imfile)

    landmarks = face_detection(image)
    masks = build_mask(landmarks[0])

    fig = plt.figure(figsize=(10,10))
    plt.imshow(image[:, :, [2, 1, 0]]) # BGR

    for mask in masks:
        plt.fill(mask[:,0], mask[:,1], color=np.array([144, 202, 249])/255 )
        plt.fill(mask[:,0], mask[:,1], color=np.array([144, 202, 249])/300 )

    plt.axis('off')
    plt.savefig(output_file, fig=fig, format='jpeg')
    
    return {
        'image': image[:, :, [2,1,0]], # return as RGB
        'masks': masks,
        'landmarks': landmarks
    }
