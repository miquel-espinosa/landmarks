import numpy as np
import os

from numpy.ma.core import masked, masked_array
import constants
import pickle as pkl
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import re
from sklearn.model_selection import LeaveOneOut

# sqr_sum = lambda arr: np.sqrt(np.sum(arr**2))

# def euclidean(arr1,arr2):
#     return sqr_sum(arr1-arr2)

def load_data():
    with open("data-3-channels.pkl", 'rb') as file1, open("labels-3-channels.pkl", "rb") as file2:
        data = pkl.load(file1)
        labels = pkl.load(file2)
    return data, labels


def get_avg_hist(data, labels, label):
    mask = labels==label
    return data[mask].mean(0)


def get_hist_normalized(img):
    img_channels = cv2.split(img)
    height = len(img)
    width = len(img[0])
    
    hist_b = cv2.calcHist([img_channels[0]], [0], None, [256], [0, 256])
    hist_g = cv2.calcHist([img_channels[1]], [0], None, [256], [0, 256])
    hist_r = cv2.calcHist([img_channels[2]], [0], None, [256], [0, 256])

    hist = np.hstack((hist_b, hist_g, hist_r)).ravel()
    normalized = hist / hist.sum()
    return normalized

def get_image_and_hist(path, name):
    img = cv2.imread(os.path.join(path,name))
    normalized = get_hist_normalized(img)
    return img, normalized

def get_prediction(img,dataset,labels):
    distances = np.sqrt(np.sum((dataset-img)**2,axis=1)) # Compute euclidean distances
    dist_sorted, labels_sorted = zip(*sorted(zip(distances,labels))) # sorted smaller to bigger

    if constants.ALGORITMO == "qnn":
        most_frequent = np.argmax(labels_sorted[:constants.Q_VALUE])
        return labels_sorted[most_frequent]

    elif constants.ALGORITMO == "1nn":
        if dist_sorted[0] < constants.DELTA:  return labels_sorted[0]
        else: return "unknown"
    


def get_accuracy(X,labels):
    loo = LeaveOneOut()
    total = len(X)
    success = 0
    for train_index, test_index in loo.split(X):
        prediction = get_prediction(X[test_index][0],X[list(train_index)], labels[list(train_index)])
        if prediction == labels[test_index][0]: success +=1

    return success/total


def main():
    data, labels = load_data()
    
    # avg_hists = []
    # for i in constants.nodes:
    #     avg = get_avg_hist(data,labels,i)
    #     avg_hists.append(avg)
    #     if constants.MEAN_HIST:
    #         plt.plot(get_avg_hist(data,labels,i))
    #         plt.savefig(str("plot"+str(i)))
    #         plt.show()

    if constants.ACCURACY:
        acc = get_accuracy(data,labels)
        print("Algorithm: ", constants.ALGORITMO)
        print("Model accuracy: ", acc*100, "%")


    if constants.VIDEO:
        cap = cv2.VideoCapture(constants.BASE_DIR+"/Video.mp4")

        w, h = 1080, 1920
        fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
        fps = 60
        file_path="/home/miquel/Documents/MASTER/Robots/Landmarks/video-final.mp4"
        writer = cv2.VideoWriter(file_path, fourcc, fps, (w, h))

        success, img = cap.read()
        
        
        while success:
            
            hist = get_hist_normalized(img)
            prediction = get_prediction(hist,data,labels)
            # prediction = get_prediction(hist,avg_hists,[0,1,2,3,4,5,6,7,8])
            
            font = cv2.FONT_HERSHEY_SIMPLEX

            if prediction in constants.nodes: # If inside keys of dictionary
                text = constants.nodes[prediction]
            else: text = "Viajando"
  
            # Use putText() method for inserting text on video
            cv2.putText(img, str(text), (50, 50), font, 2, (0, 0, 0), 2, cv2.LINE_4)
            
            writer.write(img)
            cv2.imshow('video', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                writer.release()
                break

            # read next frame
            success, img = cap.read()       

        # Close writer
        writer.release()


if __name__=="__main__":
    main()
    