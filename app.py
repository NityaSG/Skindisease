import tensorflow as tf
import streamlit as st
import flask
import cv2
classes = {4: ('nv', ' melanocytic nevi'), 6: ('mel', 'melanoma'), 2 :('bkl', 'benign keratosis-like lesions'), 1:('bcc' , ' basal cell carcinoma'), 5: ('vasc', ' pyogenic granulomas and hemorrhage'), 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  3: ('df', 'dermatofibroma')}
def process(dirr):
    model = tf.keras.models.load_model('saved_model/my_model')
    
    img = cv2.imread(dirr)
    print(type(img))
   # cv2.imwrite(dirr, img)
    #window_name = 'image'
    #cv2.imshow(window_name, img)
    img = cv2.resize(img, (28, 28))
    print(type(img))
    result = model.predict(img.reshape(1, 28, 28, 3))
    max_prob = max(result[0])
    class_ind = list(result[0]).index(max_prob)
    class_name = classes[class_ind]
    print(class_name)

dirr = './HAM10000_images_part_1/ISIC_0024306.jpg'
process(dirr)