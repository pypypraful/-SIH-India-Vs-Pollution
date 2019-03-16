import torch
from torch import nn, optim
from torchvision import datasets, transforms, models
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
from PIL import Image
import urllib
import cv2

from flask import Flask
from flask import jsonify
from flask import make_response
from urllib.error import HTTPError, URLError
from flask import request
import requests

from textblob import TextBlob
import re, os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import *
from sklearn.svm import *
import pandas

model = models.densenet121(pretrained=True)

for param in model.parameters():
    param.requires_grad = False

model.classifier = nn.Sequential(nn.Linear(1024, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 4),
                                 nn.LogSoftmax(dim=1))
model.load_state_dict(torch.load('model_cifar1.pt'))

model.eval()



def image_loader(loader, image_name):
    image = Image.open(image_name)
    image = loader(image).float().cpu()
    image = torch.tensor(image, requires_grad=True)
    image = image.unsqueeze(0)
    image = image
    return image

def url_to_image(url):
	image = urllib.request.urlopen(url)
	return image

data_transforms = transforms.Compose([
    transforms.Resize(255),
    transforms.CenterCrop(224),
    transforms.ToTensor()
])
app=Flask(__name__)
@app.route('/detect', methods=['POST'])

def detect():
    message = request.get_json()
    url = message["url"]
    error=""
    detect=""

    image = url_to_image(url)

    output = model(image_loader(data_transforms, image))
    prob = nn.functional.softmax(output, dim=1)
    final_probs = prob.detach().cpu().numpy()*100

    index_of_poll = np.argmax(final_probs)
    if final_probs.max()<50:
      index_of_poll = 4

    pollution = ["air pollution" , "land pollution", "noise pollution", "water pollution", "spam" ]

    return pollution[index_of_poll]


##########################################
#########################################

def clean_tweet(tweet):
        return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())

global Classifier
global Vectorizer

from rake_nltk import Rake

# load data
data = pandas.read_csv('spam.csv', encoding='latin-1')
train_data = data[:4400] # 4400 items
test_data = data[4400:] # 1172 items

# train model
Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
Vectorizer = TfidfVectorizer()
vectorize_text = Vectorizer.fit_transform(train_data.v2)
Classifier.fit(vectorize_text, train_data.v1)


@app.route('/tag', methods=['POST'])
def text():
    try:
        data = request.get_json()

        r = Rake()

        r.extract_keywords_from_text(data["text"])



        return make_response(jsonify({"tags": r.get_ranked_phrases()}), 200)

    except HTTPError as e:
        print(e.code)
        return str(e) + 'HTTPError'
    except URLError as e:
        print(e.args)
        return str(e) + 'Url Error'




@app.route('/sentiment', methods=['POST'])
def senti():
    try:
        data = request.get_json()
        blob = TextBlob(clean_tweet(data["text"]))
        for sentence in blob.sentences:
            sent = sentence.sentiment.polarity

        return make_response(jsonify({"sentiment": sent}), 200)

    except HTTPError as e:
        print(e.code)
        return str(e) + 'HTTPError'
    except URLError as e:
        print(e.args)
        return str(e) + 'Url Error'



@app.route('/predict', methods=['POST'])
def predict():
    message = request.get_json()
    message = message["text"]
    error = ''
    predict = ''
    predict = ''

    global Classifier
    global Vectorizer
    try:
        if len(message) > 0:
          vectorize_message = Vectorizer.transform([message])
          predict = Classifier.predict(vectorize_message)[0]
    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)
    return jsonify(
              predict=predict)


@app.route('/train_spam', methods=['POST'])
def train_spam():
    data_train = request.get_json()
    data_train = data_train["data"]

    train_data = data_train[:4400] # 4400 items
    test_data = data_train[4400:] # 1172 items


    # load data
    to_train = DateFrame.from_dict(data, encoding='latin-1')


    # train model
    Classifier = OneVsRestClassifier(SVC(kernel='linear', probability=True))
    Vectorizer = TfidfVectorizer()
    vectorize_text = Vectorizer.fit_transform(train_data.v2)
    Classifier.fit(vectorize_text, train_data.v1)

    try:
        if len(message) > 0:
          vectorize_message = Vectorizer.transform([message])
          predict = Classifier.predict(vectorize_message)[0]
    except BaseException as inst:
        error = str(type(inst).__name__) + ' ' + str(inst)
    return jsonify(
              predict=predict)
###########################################################
###########################################################


@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not Found'}), 404)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port='8800', debug=True)
