import cv2
import tensorflow as tf
import os
import time

labels = ['t pose', 'adho mukha svanasana', 'adho mukha vriksasana', 
'agnistambhasana', 'ananda balasana', 'anantasana', 'anjaneyasana', 'ardha bhekasana', 'ardha chandrasana',
'ardha matsyendrasana', 'ardha pincha mayurasana', 'ardha uttanasana', 'ashtanga namaskara', 'astavakrasana', 
'baddha konasana', 'bakasana', 'balasana', 'bhairavasana', 'bharadvajasana i', 'bhekasana', 'bhujangasana', 
'bhujapidasana', 'bitilasana', 'camatkarasana','warrior II pose', 'chakravakasana', 'chaturanga dandasana', 'dandasana', 
'dhanurasana', 'durvasasana', 'dwi pada viparita dandasana', 'eka pada koundinyanasana i', 
'eka pada koundinyanasana ii', 'utthita trikonasana', 'vajrasana', 'vasisthasana', 'viparita karani', 
'virabhadrasana i', 'virabhadrasana ii', 'virabhadrasana iii', 'virasana', 'vriksasana', 'vrischikasana', 
'yoganidrasana', 'tree pose']

width = 224
height = 224
dim = (width, height)

cam = cv2.VideoCapture(0)
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

timer = 0
poses = []
total_cals_burned = 0

def calculate_calories_burnt(pred):
    num_extended_limbs = sum([int(limb) for limb in pred.limbs])

    left_knee = pred.keypoints.LEFT_KNEE
    left_hip = pred.keypoints.LEFT_HIP
    right_knee = pred.keypoints.RIGHT_KNEE
    right_hip = pred.keypoints.RIGHT_HIP
    left_leg_distance = abs(left_knee.y - left_hip.y)
    right_leg_distance = abs(right_knee.y - right_hip.y)
    avg_leg_distance = (left_leg_distance + right_leg_distance) / 2

    calories_burnt = (num_extended_limbs * 5) + (avg_leg_distance * 0.1)

    return calories_burnt

while True:
    timer += 1
    time.sleep(0.5)
    result, image = cam.read()
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    img = tf.keras.utils.img_to_array(resized_image)
    final_img = img.reshape(1, 224, 224, 3)
    
    tflite_size = os.path.getsize('model.tflite') / 1048576
    tflite_model_path = 'model.tflite'

    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # print(input_details)
    # print(output_details)

    input_data = final_img
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data_tflite = interpreter.get_tensor(output_details[0]['index'])

    pred = output_data_tflite.argmax()

    if(timer % 5 == 0 and pred.pose != 'Unknown Pose'):
        poses.append(pred.pose)

    cals_burned = calculate_calories_burnt(pred)

    color = (0, 0, 255)
    if(pred.pose != 'Unkown Pose'):
        color = (0, 255, 0)
        total_cals_burned += cals_burned

    
    cv2.putText(img, pred.pose, (10, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.putText(img, 'Accuracy - ' + str(pred.accuracy_val), (10, 70),
                cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.putText(img, 'Total calories burned - ' + str(total_cals_burned), (650, 30),
                cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    cv2.putText(img, 'Total correct poses done - ' + str(len(poses)), (550, 70),
                cv2.FONT_HERSHEY_PLAIN, 2, color, 2)
    
    time.sleep(1)
