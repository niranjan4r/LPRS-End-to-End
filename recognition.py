import argparse
import os
import pandas as pd
import editdistance
import numpy as np
from skimage.transform import resize
from skimage import measure
from skimage.measure import regionprops
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import cv2
from utils.validate_lp import validateLP
from skimage.filters import threshold_otsu
from skimage.filters import threshold_local
from sklearn.metrics import accuracy_score
import pickle
import warnings
# warnings.filterwarnings("ignore")

#### argument parsing ####
parser = argparse.ArgumentParser()
parser.add_argument('--dataPath', required=True, help='path to training dataset')
parser.add_argument('--savePath', required=True, help='path to save results')
parser.add_argument('--modelPath', required=True, help='path to trained detection model')
# parser.add_argument('--ctcDecoder', type=str, default='bestPath', 
#                     choices=['bestPath', 'beamSearch'],
#                     help='method for decoding ctc outputs')

# parser.add_argument('--normalise', type=bool, default=False, 
#                     help='set true to normalise posterior probability.')

opt = parser.parse_args()
print(opt)

if not os.path.exists(opt.savePath):
    os.makedirs(opt.savePath)


#### load model ####
# lpr = alpr.AutoLPR(decoder=opt.ctcDecoder, normalise=opt.normalise)
# lpr.load(crnn_path=opt.crnnPath)
filename = './model_recognition.sav'
model = pickle.load(open(filename, 'rb'))

#### test performance ####
# result = pd.DataFrame([], columns=['path', 'gTruth', 'pred', 'editDistance'])

total_images = len(os.listdir(opt.dataPath))
i = 2
Dict = {}
while (i <= total_images):
    file = "output" + str(i) + ".jpg"
    img = cv2.imread(opt.dataPath + '/' + file)
    V = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))[2]
    T = threshold_local(V, 29, offset=15, method="gaussian")
    thresh = (V > T).astype("uint8") * 255
    binary_image = cv2.bitwise_not(thresh)
    # ground truth
    filename, file_extension = os.path.splitext(file)
    gt = filename.split('_')[-1]
    
    # prediction
    plate_like_objects = []
    plate_like_objects.append(binary_image)

    license_plate = plate_like_objects[0]

    labelled_plate = measure.label(license_plate)

    # fig, ax1 = plt.subplots(1)
    # ax1.imshow(license_plate, cmap="gray")

    # the next two lines is based on the assumptions that the width of
    # a license plate should be between 5% and 15% of the license plate,
    # and height should be between 35% and 60%
    # this will eliminate some
    character_dimensions = (0.35*license_plate.shape[0], 0.80*license_plate.shape[0], 0.02*license_plate.shape[1], 0.15*license_plate.shape[1])
    min_height, max_height, min_width, max_width = character_dimensions

    characters = []
    counter=0
    column_list = []
    for regions in regionprops(labelled_plate):
        y0, x0, y1, x1 = regions.bbox
        region_height = y1 - y0
        region_width = x1 - x0

        if region_height > min_height and region_height < max_height and region_width > min_width and region_width < max_width:
            roi = license_plate[y0:y1, x0:x1]
            # draw a red bordered rectangle over the character.
            # rect_border = patches.Rectangle((x0, y0), x1 - x0, y1 - y0, edgecolor="red",
            #                               linewidth=2, fill=False)
            # ax1.add_patch(rect_border)

            # resize the characters to 20X20 and then append each character into the characters list
            resized_char = resize(roi, (20, 20))
            characters.append(resized_char)

            # this is just to keep track of the arrangement of the characters
            column_list.append(x0)
    # print(characters)
    # plt.show()


    classification_result = []
    for each_character in characters:
        # converts it to a 1D array
        each_character = each_character.reshape(1, -1);
        res = model.predict(each_character)
        classification_result.append(res)

    # print('Classification result')
    # print(classification_result)

    plate_string = ''
    for eachPredict in classification_result:
        plate_string += eachPredict[0]

    # print('Predicted license plate')
    # print(plate_string)

    # it's possible the characters are wrongly arranged
    # since that's a possibility, the column_list will be
    # used to sort the letters in the right order

    column_list_copy = column_list[:]
    column_list.sort()
    rightplate_string = ''
    for each in column_list:
        rightplate_string += plate_string[column_list_copy.index(each)]

    # print(rightplate_string)
    if (rightplate_string in Dict):
            Dict[rightplate_string] += 1
    elif (rightplate_string != ""):
        Dict[rightplate_string] = 1

    if ((i % 15 == 0 or i == total_images) and bool(Dict)):
        Keymax = max(zip(Dict.values(), Dict.keys()))[1]
        if (len(Keymax) > 3):
          Keymax = validateLP(Keymax)
          print("LP Number: " + Keymax)
        Dict = {}
    i += 1
        # distance
#         dist = editdistance.eval(gt, rightplate_string)
        
#         result = result.append({'path': file,
#                                 'gTruth': gt,
#                                 'pred': rightplate_string,
#                                 'editDistance': dist}, ignore_index=True)
        
# #### print and save results ####
# # print("Accuracy:", accuracy_score(result.gTruth, result.pred))
# print('\n')
# print("Edit Distance Distribution")
# print(result.editDistance.value_counts(sort=False))

# result = result.sort_values("editDistance", ascending=False).reset_index(drop=True)
# result.to_csv(os.path.join(opt.savePath, 'result.csv'),
#               index=False)