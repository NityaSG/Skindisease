import tensorflow as tf
import streamlit as st
from PIL import Image, ImageOps
import numpy as np
classes = {4: ('nv', ' melanocytic nevi'), 6: ('mel', 'melanoma'), 2 :('bkl', 'benign keratosis-like lesions'), 1:('bcc' , ' basal cell carcinoma'), 5: ('vasc', ' pyogenic granulomas and hemorrhage'), 0: ('akiec', 'Actinic keratoses and intraepithelial carcinomae'),  3: ('df', 'dermatofibroma')}
def process(model,img):
    imag = ImageOps.fit(img, (28,28), Image.ANTIALIAS)
    imag = np.asarray(imag)
    result = model.predict(imag.reshape(1, 28, 28, 3))
    max_prob = max(result[0])
    class_ind = list(result[0]).index(max_prob)
    class_name = classes[class_ind]
    #st.subheader(class_name[1])
    return(class_name[1])

def info(output):
    if output == "melanocytic nevi":
        st.text("Melanocytic nevi are usually noncancerous growths on the skin that are formed by a cluster of melanocytes (cells that make a substance called melanin, which gives color to skin and eyes). They are usually dark and may be raised from the skin. They can be brown, tan, black, red, blue or pink and can be smooth, wrinkled, flat or raised. They may have hair growing from them")
model = tf.keras.models.load_model('saved_model/my_model')
co1,co2,co3=st.columns([1,2,1])
co2.image('1111.png',width=300) 
    #st.subheader("About")
    #st.text_area("This model uses Deep learning to classify the skin disease in the given image. The model's architecture is developed using deep Convolutional Neural network.")
#st.markdown('<style> .appview-container .main .block-container{{ padding-top: {padding_top}rem;    }}</style>',unsafe_allow_html=True)
st.markdown('<style> }} .appview-container .main .block-container{{ max-width: {percentage_width_main}%; padding-top: {1}rem; padding-right: {1}rem; padding-left: {1}rem; padding-bottom: {1}rem; }} .uploadedFile {{display: none}} footer {{visibility: hidden;}} </style>',unsafe_allow_html=True)
 
#st.title("Skin disease prediction")
#c1.text_area("Original Data Source Original Challenge: https://challenge2018.isic-archive.com https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T [1] Noel Codella, Veronica Rotemberg, Philipp Tschandl, M. Emre Celebi, Stephen Dusza, David Gutman, Brian Helba, Aadi Kalloo, Konstantinos Liopyris, Michael Marchetti, Harald Kittler, Allan Halpern: Skin Lesion Analysis Toward Melanoma Detection 2018: A Challenge Hosted by the International Skin Imaging Collaboration (ISIC), 2018; https://arxiv.org/abs/1902.03368")
st.header("skin disease classification")
st.caption("This model uses Deep learning to classify the skin disease in the given image. The model's architecture is developed using deep Convolutional Neural network.")
c1,c2=st.columns([2,2])
file = c1.file_uploader("Please upload an image file", type=["jpg", "png"])
if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    c2.image(image, use_column_width=True)
    output=process(model,image)
    c1.subheader("the Predicted disease is : ")
    c1.text(output)
    if len(output) == 17:
        st.subheader(output)
        st.caption("Melanocytic nevi are usually noncancerous growths on the skin that are formed by a cluster of melanocytes (cells that make a substance called melanin, which gives color to skin and eyes). They are usually dark and may be raised from the skin. They can be brown, tan, black, red, blue or pink and can be smooth, wrinkled, flat or raised. They may have hair growing from them")
        st.subheader("What are the complications of melanocytic naevi?")
        st.caption(" People worry about their moles because they have heard about melanoma, a malignant proliferation of melanocytes that is the most common reason for death from skin cancer. At first, melanoma may look similar to a harmless melanocytic naevus, but in time it becomes more disordered in structure and tends to enlarge. People with a greater number of naevi have a higher risk of developing melanoma than those with few naevi, especially if they have over 100 of them. Melanocytic naevi sometimes change for other reasons than melanoma, for example following sun exposure or during pregnancy. They can enlarge, regress or involute (disappear). A Meyerson naevus is itchy and dry because it is surrounded by eczema. A Sutton or halo naevus is surrounded by a white patch and fades away over several years A recurrent naevus is one that appears in a scar following surgical removal of a melanocytic naevus — this may have an odd shape.")
        st.subheader("Acquired melanocytic naevi")
        st.image('mole.jpg')
        a,s,d,f=st.columns([1,1,1,1])
        a.subheader("Café au lait macule")
        a.caption("Café au lait macule is a flat brown patch.")
        a.image("m1.jpg")
        s.subheader("Speckled lentiginous naevus")
        s.caption("Speckled lentiginous naevus is a flat brown patch with darker spots.")
        s.image("m2.jpg")
        d.subheader("Naevus of Ota")
        d.caption("Naevus of Ota is a bluish brown mark around forehead, eye and cheek.")
        f.subheader("Mongolian spot")
        f.caption("Mongolian spot is a large bluish mark most often seen on buttocks of a newborn.")

    info(output)

