import tensorflow as tf
import numpy as np
import cv2
import pathlib

#edit path to model
interpreter = tf.lite.Interpreter(model_path="/home/ariel/ADUK/cvml/OBJECTRON/tensorflow_model/object_detection_3d_cup.tflite")

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print('input dets: ', input_details)
print('output dets: ', output_details)

interpreter.allocate_tensors()

def im_square(image):
    h = image.shape[0]
    w = image.shape[1]
    side = min(h, w)
    crop_dist = int((max(h, w) - side)/2)
    if h > w:
        crop_img = image[crop_dist:h - crop_dist, 0:side]
    else:
        crop_img = image[0:side, crop_dist:w - crop_dist]
    
    return crop_img


def multiple_by_value(array, value):
    for i in range(len(array)):
        array[i-1] = array[i-1] * value
    return array

def draw_box(img, rects):
    cv2.circle(img, (int(rects[0][0]), int(rects[0][1])), radius = 0, color=(0, 255, 0), thickness=5)

    img = cv2.line(img, (int(rects[0][2]), int(rects[0][3])), (int(rects[0][4]), int(rects[0][5])), color=(0, 0, 255), thickness=1)
    img = cv2.line(img, (int(rects[0][4]), int(rects[0][5])), (int(rects[0][12]), int(rects[0][13])), color=(0, 0, 255), thickness=1)
    img = cv2.line(img, (int(rects[0][12]), int(rects[0][13])), (int(rects[0][10]), int(rects[0][11])), color=(0, 0, 255), thickness=1)
    img = cv2.line(img, (int(rects[0][2]), int(rects[0][3])), (int(rects[0][10]), int(rects[0][11])), color=(0, 0, 255), thickness=1)

    img = cv2.line(img, (int(rects[0][6]), int(rects[0][7])), (int(rects[0][8]), int(rects[0][9])), color=(0, 0, 255), thickness=1)
    img = cv2.line(img, (int(rects[0][8]), int(rects[0][9])), (int(rects[0][16]), int(rects[0][17])), color=(0, 0, 255), thickness=1)
    img = cv2.line(img, (int(rects[0][16]), int(rects[0][17])), (int(rects[0][14]), int(rects[0][15])), color=(0, 0, 255), thickness=1)
    img = cv2.line(img, (int(rects[0][14]), int(rects[0][15])), (int(rects[0][6]), int(rects[0][7])), color=(0, 0, 255), thickness=1)

    img = cv2.line(img, (int(rects[0][2]), int(rects[0][3])), (int(rects[0][6]), int(rects[0][7])), color=(0, 0, 255), thickness=1)
    img = cv2.line(img, (int(rects[0][4]), int(rects[0][5])), (int(rects[0][8]), int(rects[0][9])), color=(0, 0, 255), thickness=1)
    img = cv2.line(img, (int(rects[0][12]), int(rects[0][13])), (int(rects[0][16]), int(rects[0][17])), color=(0, 0, 255), thickness=1)
    img = cv2.line(img, (int(rects[0][10]), int(rects[0][11])), (int(rects[0][14]), int(rects[0][15])), color=(0, 0, 255), thickness=1)

    img = cv2.line(img, (int(rects[0][4]), int(rects[0][5])), (int(rects[0][10]), int(rects[0][11])), color=(0, 0, 255), thickness=1)
    img = cv2.line(img, (int(rects[0][2]), int(rects[0][3])), (int(rects[0][12]), int(rects[0][13])), color=(0, 0, 255), thickness=1)

    i=2
    while i < 18:
        cv2.circle(img, (int(rects[0][i]), int(rects[0][i+1])), radius = 0, color=(255, 255, 0), thickness=5)
        i+=2
    
    return img


img = cv2.imread('/home/ariel/ADUK/cvml/OBJECTRON/tensorflow_model/images/6.jpg')
if img.shape[0] != img.shape[1]:
    img = im_square (img)
im_square(img)
new_img = cv2.resize(img, (224, 224))
new_img = new_img.astype(np.float32)
new_img /= 255.0

interpreter.set_tensor(input_details[0]['index'], [new_img])

interpreter.invoke()
rects = interpreter.get_tensor(output_details[0]['index'])
scores = interpreter.get_tensor(output_details[1]['index'])

if scores[0] > 0.5:
    normal_img = draw_box(img.copy(), multiple_by_value(rects, (img.shape[0]/224)))
    cv2.putText(normal_img, 'score: ' + str((scores[0])*100) + '%', (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)

cv2.imshow('normal img: ', normal_img)

print("Rectangles are: ", rects)
print("Scores are: ", scores)
cv2.waitKey(0)
