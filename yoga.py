import cv2
import tensorflow as tf
import os
import time

labels = ['t pose', 'warrior II pose', 'tree pose', 'adho mukha svanasana', 'adho mukha vriksasana', 'agnistambhasana', 'ananda balasana', 'anantasana', 'anjaneyasana', 'ardha bhekasana', 'ardha chandrasana', 'ardha matsyendrasana', 'ardha pincha mayurasana', 'ardha uttanasana', 'ashtanga namaskara', 'astavakrasana', 'baddha konasana', 'bakasana', 'balasana', 'bhairavasana', 'bharadvajasana i', 'bhekasana', 'bhujangasana', 'bhujapidasana', 'bitilasana', 'camatkarasana', 'chakravakasana', 'chaturanga dandasana', 'dandasana', 'dhanurasana', 'durvasasana', 'dwi pada viparita dandasana', 'eka pada koundinyanasana i', 'eka pada koundinyanasana ii', 'eka pada rajakapotasana', 'eka pada rajakapotasana ii', 'ganda bherundasana', 'garbha pindasana', 'garudasana', 'gomukhasana', 'halasana', 'hanumanasana', 'janu sirsasana', 'kapotasana', 'krounchasana', 'kurmasana', 'lolasana', 'makara adho mukha svanasana', 'makarasana', 'malasana', 'marichyasana i', 'marichyasana iii', 'marjaryasana', 'matsyasana', 'mayurasana', 'natarajasana', 'padangusthasana', 'padmasana', 'parighasana', 'paripurna navasana', 'parivrtta janu sirsasana', 'parivrtta parsvakonasana', 'parivrtta trikonasana', 'parsva bakasana', 'parsvottanasana', 'pasasana', 'paschimottanasana', 'phalakasana', 'pincha mayurasana', 'prasarita padottanasana', 'purvottanasana', 'salabhasana', 'salamba bhujangasana', 'salamba sarvangasana', 'salamba sirsasana', 'savasana', 'setu bandha sarvangasana', 'simhasana', 'sukhasana', 'supta baddha konasana', 'supta matsyendrasana', 'supta padangusthasana', 'supta virasana', 'tadasana', 'tittibhasana', 'tolasana', 'tulasana', 'upavistha konasana', 'urdhva dhanurasana', 'urdhva hastasana', 'urdhva mukha svanasana', 'urdhva prasarita eka padasana', 'ustrasana', 'utkatasana', 'uttana shishosana', 'uttanasana', 'utthita ashwa sanchalanasana', 'utthita hasta padangustasana', 'utthita parsvakonasana', 'utthita trikonasana', 'vajrasana', 'vasisthasana', 'viparita karani', 'virabhadrasana i', 'virabhadrasana ii', 'virabhadrasana iii', 'virasana', 'vriksasana', 'vrischikasana', 'yoganidrasana']

width = 224
height = 224
dim = (width, height)

cam = cv2.VideoCapture(0)
cv2.namedWindow('Pose Classification', cv2.WINDOW_NORMAL)

while True:
    result, image = cam.read()
    resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    img = tf.keras.utils.img_to_array(resized_image)
    print(img.shape)
    final_img = img.reshape(-1, 224, 224, 3)
    tflite_size = os.path.getsize('model.tflite') / 1048576
    tflite_model_path = 'model.tflite'
    interpreter = tf.lite.Interpreter(tflite_model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_data = final_img
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data_tflite = interpreter.get_tensor(output_details[0]['index'])
    final_prediction = output_data_tflite.argmax()
    pred = labels[final_prediction]
    print(pred)
    time.sleep(1)
