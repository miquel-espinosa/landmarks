import cv2
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import pickle as pkl

import constants

def get_images_histograms(folder):
    hists = []
    images = []
    for filename in os.listdir(folder):
        img = cv2.imread(os.path.join(folder,filename))
        img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        img_channels = cv2.split(img)
        
        height = len(img)
        width = len(img[0])
        
        hist_b = cv2.calcHist([img_channels[0]], [0], None, [256], [0, 256])
        hist_g = cv2.calcHist([img_channels[1]], [0], None, [256], [0, 256])
        hist_r = cv2.calcHist([img_channels[2]], [0], None, [256], [0, 256])


        hist = np.hstack((hist_b, hist_g, hist_r)).ravel()
        normalized = hist / hist.sum()
        # normalized = hist/(width*height) # Normalized histogram
        # hists.append([normalized,index_class])
        hists.append(normalized)
        images.append(img)
            

    return images, np.array(hists)



for folder in os.listdir(constants.IMG_DIR):
    # If we have the value, search the key in the constants.nodes dictionary
    label = list(constants.nodes.keys())[list(constants.nodes.values()).index(folder)]

    images, histograms = get_images_histograms(os.path.join(constants.IMG_DIR,folder))
    labels = np.full((len(histograms),), label)
    
    if label == 0:
        final_histograms = histograms
        final_labels = labels
    else:
        final_histograms = np.vstack((final_histograms,histograms))
        final_labels = np.hstack((final_labels,labels))


with open("data-3-channels.pkl", "wb") as file1, open("labels-3-channels.pkl", "wb") as file2:
    pkl.dump(final_histograms,file1)
    pkl.dump(final_labels,file2)
