from flask import Flask,render_template,request
import tensorflow as tf 
import keras
from keras.preprocessing.image import load_img, img_to_array
import numpy as np 
import skimage as sk 
model=keras.models.load_model("../models/ version_1")

main_app=Flask(__name__)
@main_app.route("/",methods=["GET"])
def homepage():
    return render_template("home.html",pagetitle="home")
@main_app.route("/",methods=["POST"])
def predict():
    imagee= request.files["imagefile"]
    path= './images/'+imagee.filename
    imagee.save(path)
    pre_image= load_img(path,target_size=(256,256))
    image=img_to_array(pre_image).reshape(1,256,256,3)
    class_names=["Early blight","Late blight","Healthy"]
    prediction=model.predict(image)
    predicted_class= class_names[np.argmax(prediction)]
    accuracy= str(round(np.max(prediction)*100,2))+ " %"
    show='images/'+ imagee.filename
    return render_template("home.html",pagetitle="home",classname=predicted_class,accuracy=accuracy,image_path=show)
    
if __name__ =="__main__":
    main_app.run(debug=True)