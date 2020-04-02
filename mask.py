import numpy as np
import dlib
import cv2
import argparse
import os
from image_utility import save_image, generate_random_color, draw_border
from imutils import face_utils
import scipy
from scipy.spatial import Delaunay

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


def add_auxilliary_landmarks(lm):
    right_cheekbone = np.round(0.55*lm[25] + 0.45*lm[11])
    below_right_cheekbone = np.round(0.5*right_cheekbone + 0.5*lm[54])
    left_cheekbone = np.round(0.55*lm[18] + 0.45*lm[5])
    below_left_cheekbone = np.round(0.5*left_cheekbone + 0.5*lm[48])
    right_upper_nostril = np.round(0.7*lm[31] + 0.3*lm[41])
    left_upper_nostril = np.round(0.7*lm[35] + 0.3*lm[46])
    
    inner_jaw_line_1 = 0.75*lm[2:15] + 0.25*lm[30]
    inner_jaw_line_2 = 0.9*lm[2:15] + 0.1*lm[30]
    
    aux_lm = np.vstack([
        lm,
        right_cheekbone,
        below_right_cheekbone,
        left_cheekbone,
        below_left_cheekbone,
        right_upper_nostril,
        left_upper_nostril,
        inner_jaw_line_1,
        inner_jaw_line_2
    ])
    
    return aux_lm

def subset_landmark_points(lm):
    lm_subset = np.vstack([
        lm[2:15,:],
        #lm[0:17,:],
        #lm[28],
        lm[29],
        lm[30],
        lm[31],
        lm[33],
        lm[35],
        lm[48:60],
        lm[68:,:],# all auxiliary points
    ])
    
    # Add random points to improve resolution
    rp = np.random.rand(1000, 2)
    rp[:,0] = rp[:,0]*(np.max(lm[:,0]) - np.min(lm[:,0])) + np.min(lm[:,0])
    rp[:,1] = rp[:,1]*(np.max(lm[:,1]) - np.min(lm[:,1])) + np.min(lm[:,1])
    
    tri = Delaunay(lm_subset)
    mask = tri.find_simplex(rp)>=0
    lm_subset = np.vstack([lm_subset, rp[mask]])
    
    
    return lm_subset

def get_mid_line_eqns(lm):
    # Assumes unaltered landmarks from dlib.
    delta = lm[27] - lm[8]
    m = delta[1]/delta[0]
    b = lm[8][1] - m*lm[8][0]
    
    # orthogonal line, running through middle of face.
    mo = -1/m
    bo = lm[27][1] - mo*lm[27][0]
    
    return m,b,mo,bo

def distance_of_point_to_line(point, m, b):
    # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line
    # a = m, b = -1, c = b
    
    return np.abs(m*point[:,0] + (-1)*point[:,1] + b)/(np.sqrt(m**2 + (-1)**2))



def calculate_delauney_mask_triangles(face_landmarks):
    # face_landmarks = landmark points for a single face.
    
    # Add auxiliary landmark points
    aux_lm = add_auxilliary_landmarks(face_landmarks)
    
    # Subset to those just for the mask.
    # IMPORTANT: everytime this function changes, 
    # like when we update the triangulating points,
    # the semantic centroid dist mat needs to be recomputed.
    sub_aux_lm = subset_landmark_points(aux_lm)
    
    # face midline equations based on all landmarks.
    midline_eqns = get_mid_line_eqns(face_landmarks)
    
    # Delaunay triangulation
    tri = Delaunay(sub_aux_lm)
    tric = sub_aux_lm[tri.simplices] # Triangle coords: num triangles x triangle corners x 2D.
    centroids = np.mean(tric, axis=1)
    
    
    # Compute distance matrices based on distance from
    # vertical and horizontal midline (these should be semantically related)
    # as well as just plain euclidean distance (nearby patches should be similar)
    v_mid_dists = distance_of_point_to_line(centroids, midline_eqns[0], midline_eqns[1])
    h_mid_dists = distance_of_point_to_line(centroids, midline_eqns[2], midline_eqns[3])

    v_pwd = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(v_mid_dists.reshape((-1,1))))
    h_pwd = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(h_mid_dists.reshape((-1,1))))
    pwd = scipy.spatial.distance.squareform(
        scipy.spatial.distance.pdist(centroids))

    # weighted average
    weights = np.array([1,1,0.05])#np.array([1,0.5,0.05]) ## CHANGE THESE TO GET DESIRED BEHAVIOR
    pwds = [v_pwd, h_pwd, pwd]
    combined_pwd = np.zeros(pwd.shape)
    for i in range(len(weights)):
        combined_pwd += weights[i]*pwds[i]

    combined_pwd = combined_pwd/np.sum(weights)
    
    assert combined_pwd.shape[0] == centroids.shape[0]
    
    return {
        'delauney': tri,
        'triangles': tric,
        'triangle_centroids': centroids,
        'midline_eqns': midline_eqns,
        'sub_aux_landmarks': sub_aux_lm,
        'aux_landmarks': aux_lm,
        'landmarks': face_landmarks,
        'combined_pwd': combined_pwd,
        'simple_pwd': pwd,
        'vertical_pwd': v_pwd,
        'horizonal_pwd': h_pwd
    }


### COLOR CALCULATION ###
def get_brightness(src):
    # remember image is in BGR order.
    R = src[:,:,2]
    G = src[:,:,1]
    B = src[:,:,0]
    
    y = (0.375*R + 0.5*G + 0.125*B)

    return (y - np.min(y))/(np.max(y) - np.min(y))

def calc_patch_brightness(Y, coord, rel_patch_size=0.05): ## PATCH SIZE
    patch_hw = int(np.round(Y.shape[0]*rel_patch_size))
        
    # NOTE IMAGE ROWS ARE THE Y-AXIS
    # IMAGE COLS ARE THE X-AXIS
    # "SWAP"
    c_beg = int(np.maximum(coord[0] - patch_hw, 0))
    c_end = int(np.minimum(coord[0] + patch_hw, Y.shape[1]))
    
    r_beg = int(np.maximum(coord[1] - patch_hw, 0))
    r_end = int(np.minimum(coord[1] + patch_hw, Y.shape[0]))
        
    patch = Y[r_beg:r_end][:,c_beg:c_end]
    
    return np.mean(patch)

def calc_color_triangles(face_tri_res, image, K, base_color, neigh_w=0.4):
    centroids = face_tri_res['triangle_centroids']
    tri_pwd = face_tri_res['combined_pwd']

    # The color of a triangle should be a combination of the shade
    # underneath it, plus the shade of its semantic neighbors.
    colors = np.vstack([base_color]*centroids.shape[0])

    Y = get_brightness(image)
    centroid_brightnesses = np.array([calc_patch_brightness(Y, c) for c in centroids])

    brightness_delta = np.zeros(centroids.shape[0])
    for i in range(centroids.shape[0]):
        c = colors[i]

        # Get nearest neighbors.
        # sort distances smallest to largest
        sidx_knn = np.argsort(tri_pwd[i])[1:K+1] # skip first one, which is ourself.

        this_brightness = centroid_brightnesses[i]
        neigh_brightness = centroid_brightnesses[sidx_knn]

        brightness_delta[i] = ((1-neigh_w)*this_brightness - neigh_w*np.mean(neigh_brightness)) # 0.6, 0.4

    # Now we need to infer the scale and mean of the brightness deltas.
    Y = get_brightness(image)
    sp = np.random.rand(10000,2)
    sp[:,0] *= Y.shape[1]
    sp[:,1] *= Y.shape[0]

    mask = face_tri_res['delauney'].find_simplex(sp)>=0
    sp = sp[mask].astype(int)
    
#     plt.plot(sp[:,0], sp[:,1], '.k')
#     plt.plot(face_tri_res['sub_aux_landmarks'][:,0], face_tri_res['sub_aux_landmarks'][:,1], '.r')
#     plt.show()

    y_sub = Y[sp[:,1],:][:,sp[:,0]].reshape(-1)
    y_sub_deltas = (y_sub - np.mean(y_sub))
    y_sub_delta_mu = np.mean(y_sub_deltas)
    y_sub_delta_std = np.std(y_sub_deltas)

    brightness_delta_z = (brightness_delta - np.mean(brightness_delta))/np.std(brightness_delta)
    brightness_delta = brightness_delta_z*y_sub_delta_std + y_sub_delta_mu
    
#     plt.hist(y_sub_deltas, bins=30, color='b', alpha=0.3, normed=True)
#     plt.hist(brightness_delta, bins=30, color='r', alpha=0.3, normed=True)
#     plt.show()

    #brightness_delta = 8*(brightness_delta - np.mean(brightness_delta)) - 0.2 # suhail
    #brightness_delta = 3*(brightness_delta - np.mean(brightness_delta)) - 0.2
    colors[:,0] += brightness_delta
    colors[:,1] += brightness_delta
    colors[:,2] += brightness_delta
    colors = np.maximum(np.minimum(colors, 1), 0)
    
    return colors




def impose_mask(imfile, output_file, color=np.array([173, 216, 230])/300):
    assert output_file[-4:] == '.jpg' or output_file[-5:] == '.jpeg'
    
    image = cv2.imread(imfile)

    landmarks = face_detection(image) # landmark detection for all faces.

    # For every face
    #    calculate delauney triangulation.
    #    calculate the color of each triangle.
    tri_res = []
    for i,lm in enumerate(landmarks):
        tr = calculate_delauney_mask_triangles(landmarks[i])    
        tr['colors'] = calc_color_triangles(tr, image, K=70, base_color=color)
        tri_res.append(tr)
        
    
    # Render image
    fig = plt.figure(figsize=(10,10))
    plt.imshow(image[:, :, [2, 1, 0]]) # BGR

    for i,tr in enumerate(tri_res): # for every face
        for j, t in enumerate(tr['triangles']): # for every triangle 
            plt.fill(t[:,0], t[:,1], color=tr['colors'][j], alpha=1)

    plt.axis('off')
    plt.savefig(output_file, fig=fig, format='jpeg')

## DEPRECATED

# def build_mask(lm):
#     left_temple = 0.5*lm[0] + 0.5*lm[17]
#     right_temple = 0.5*lm[26] + 0.5*lm[16]
    
#     left_jaw = lm[np.arange(0,8+1),:]
# #     left_jaw = np.vstack([
# #         lm[0],
# #         0.8*lm[0] + 0.2*lm[1],
# #         0.8*left_temple + 0.2*lm[4],
# #         0.2*left_temple + 0.8*lm[4],
# #         0.2*lm[2] + 0.8*lm[3],
# #         lm[3],
# #         0.2*left_temple + 0.8*lm[4],
# #         lm[5:9,:]
# #     ])
    
    
#     right_jaw = lm[np.arange(9,17), :]
# #     right_jaw = np.flipud(np.vstack([
# #         lm[16],
# #         0.5*lm[16] + 0.5*lm[15],
# #         0.7*right_temple + 0.3*lm[12],
# #         0.3*right_temple + 0.7*lm[12],
# #         0.5*lm[14] + 0.5*lm[13],
# #         lm[13],
# #         0.2*right_temple + 0.8*lm[12],
# #         lm[[11,10, 9, 8],:]
# #     ]))

#     right_cheekbone = [0.6*lm[25] + 0.4*lm[11],
#                         0.4*lm[23] + 0.6*lm[53],
#                        0.3*lm[22] + 0.7*lm[35]]

#     left_cheekbone = [0.3*lm[21] + 0.7*lm[31],
#                         0.4*lm[20] + 0.6*lm[49],
#                       0.6*lm[18] + 0.4*lm[4]]
#     nose = lm[[29, 30]]
#     lips = lm[[51,62, 66,57]]
#     chin = lm[8]


#     right_mask = np.vstack([
#         right_jaw,
#         right_cheekbone,
#         nose,
#         lips,
#         chin
#     ])

#     left_mask = np.vstack([
#         chin,
#         np.flipud(lips),
#         np.flipud(nose),
#         left_cheekbone,
#         left_jaw
#     ])
    
#     return left_mask, right_mask




## DEPRECATED
# def impose_mask(imfile, output_file):
#     assert output_file[-4:] == '.jpg' or output_file[-5:] == '.jpeg'
    
#     image = cv2.imread(imfile)

#     landmarks = face_detection(image)
#     masks = build_mask(landmarks[0])
#     #colors = [np.array([144, 202, 249])/255, np.array([144, 202, 249])/300]
#     colors = [np.array([255, 105, 180])/255, np.array([255, 105, 180])/300]
#     assert len(colors) == len(masks)

#     fig = plt.figure(figsize=(10,10))
#     plt.imshow(image[:, :, [2, 1, 0]]) # BGR

#     for i,mask in enumerate(masks):
#         plt.fill(mask[:,0], mask[:,1], color=colors[i])

#     plt.axis('off')
#     plt.savefig(output_file, fig=fig, format='jpeg')
    
#     return {
#         'image': image[:, :, [2,1,0]], # return as RGB
#         'masks': masks,
#         'landmarks': landmarks
#     }
