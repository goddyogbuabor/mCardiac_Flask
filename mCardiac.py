from flask import Flask, render_template, request, url_for, redirect 
from recognition import Classification
from feature_extract import Feature_extraction
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import seaborn as sns
import math
import os
import glob
import io
import base64
import matplotlib.pyplot as plt

from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

app = Flask(__name__)    
app.config["UPLOAD_FOLDER"] = os.path.join('static', 'ClassifyData')
app.config["FILE_UPLOADS"] = os.path.join('static', 'TrainData')
app.config["ALLOWED_FILE_EXTENSIONS"] = ["JPEG", "PDF", "PNG", "CSV"]

def allowed_file(filename):

    if not "." in filename:
        return False

    ext = filename.rsplit(".", 1)[1]

    if ext.upper() in app.config["ALLOWED_FILE_EXTENSIONS"]:
        return True
    else:
        return False

@app.route("/upload_train_files", methods=["GET", "POST"])
def upload_train_files():
    from datetime import datetime
    if request.method == "POST":
       if request.files:
            activity = request.files["file"]
            if activity.filename == "":
                return redirect(request.url)

            if allowed_file(activity.filename):
                filename = secure_filename(activity.filename)
                activity.save(filename)
              
                return redirect(request.url)
            else:
                return redirect(request.url)
    return render_template("UploadTraining.html", success = "File Uploaded Successfully:    " )


@app.route("/upload_classify_files", methods=["GET", "POST"])
def upload_classify_files():
    from datetime import datetime
    if request.method == "POST":
        if request.files:
            activity = request.files["file"]
            if activity.filename == "":
                #print("No filename")
                return redirect(request.url)

            if allowed_file(activity.filename):
                filename = secure_filename(activity.filename)
                activity.save(filename)
                return redirect(request.url)

            else:
                return redirect(request.url)

    return render_template("uploadclassify.html", success = "File Uploaded Successfully:   ")
@app.route("/")  
def index():  
    return render_template("index.html")  

@app.route("/uploadTrain")  
def uploadTrain():  
     return render_template("UploadTraining.html")  

@app.route("/uploadClassify")  
def uploadClassify():  
    return render_template("UploadClassify.html")  

@app.route("/upload")  
def upload():  
    return render_template("UploadClassify.html")  
@app.route("/sendEmail")  
def sendEmail():  
    return render_template("Sendmail.html")  

@app.route("/CombineTrainData")  
def CombineTrainData():  
    extractF = Feature_extraction()
    extractF.join_train_data()  
    return render_template("index.html",success = "Files Combined Successfully:   ") 
@app.route("/CombineClassifyData")  
def CombineClassifyData():  
    extractF = Feature_extraction()
    extractF.join_classify_data()  
    return render_template("UploadClassify.html",success = "Files Combined Successfully:   ") 

@app.route("/train")      
def TrainAlgorithm():
    extractF = Feature_extraction()
    extractF.train_feature()
    clf = Classification()
    clf.train_algorithm()
    return render_template("uploadTraining.html", success = "Algorithm Trained Successfully:") 

@app.route("/classify")      
def ClassifyActivity():
    extract = Feature_extraction()
    extract.Classify_feature()
    clf = Classification()
    clf.classify()
    return render_template("uploadClassify.html", success = "Activity Classified Successfully:") 


@app.route("/search", methods = ["POST", "GET"])      
def SearchActivity():
        
    dataDir ='../mCardiac_flask/'
    dataset = pd.read_csv(dataDir+'ActivityData.csv')
    # Generate plot
    fig = plt.figure(figsize=(10,6))
    fig.add_axes()
    
    if request.method == "POST": 
        start_date = request.form['activity-start']
        end_date = request.form['activity-end']
        
        df = dataset[(dataset['timestamp'] >start_date) & (dataset['timestamp'] <= end_date)]
        ax = sns.scatterplot(x='timestamp', y='Activity', hue='Activity', data=df, legend=False)
        plt.gcf().autofmt_xdate()
        for label in ax.xaxis.get_ticklabels()[::2]:
                label.set_visible(False)
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    return render_template("viewdetails.html", user_image = pngImageB64String) 

@app.route("/view", methods=["Get"])
def plotView():
     
    dataDir ='../mCardiac_flask/'
    dataset = pd.read_csv(dataDir+'ActivityData.csv')
    # Generate plot
    fig = plt.figure(figsize=(10,6))
    fig.add_axes()
    dataset['ActivityC'] = dataset['Activity']
    dataset.ActivityC.replace(to_replace=dict(Walking=1, Sitting=2, Standing=3, Jogging=4), inplace=True, regex=True)
    ax= sns.scatterplot(x='timestamp', y='Activity', hue='Activity', data=dataset, legend=False)
    for label in ax.xaxis.get_ticklabels()[::2]:
            label.set_visible(False)
    plt.gcf().autofmt_xdate()
    pngImage = io.BytesIO()
    FigureCanvas(fig).print_png(pngImage)
    
    # Encode PNG image to base64 string
    pngImageB64String = "data:image/png;base64,"
    pngImageB64String += base64.b64encode(pngImage.getvalue()).decode('utf8')
    
    return render_template("viewdetails.html", user_image=pngImageB64String)


if __name__ == "__main__":  
    app.run(debug=True)  