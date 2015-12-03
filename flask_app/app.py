import json
import requests
import socket
import time
import cPickle as pickle
import pandas as pd
import re
import StringIO
from HTMLParser import HTMLParser
from flask import Flask, request, render_template
from processor import Processor

app = Flask(__name__)

## Welcome to the Email Rank flask app. This runs a persistent flask app that takes
## email input from users and returns predictions based on the models trained in the
## Ipython notebooks found in this directory

## Please refer to the 'emailRank_home.html' for the base page that is modified here

## Global variables
PORT = 8080
MODELTO = None
MODELFROM = None
PROCESSOR = None
VERBOSE = True


## Home directory, contains a page that asks for email & a submit box
@app.route('/')
def index():

    data = '''
        <h1>Welcome to Email Rank <br> </h1></h2>This tool will help you gauge the authority of your email </h2>
        <form action="/predict" method='POST' >
            <form action="/predict" method='POST' >
            <textarea name="user_input" cols="80" rows="20" >Please insert email text here...</textarea>
            <br><input type="submit" />
        '''

    return render_template('emailRank_home.html').format(data)
    

## Predict method used, receives data from the index method and returns a page with predictions
@app.route('/predict', methods=['POST'])
def predict():

    # Method to predict data and send to PSQL database
    # is automatically called based on the score POST request
    # sent by the data server
    # requires the database table to be built already

    text = request.form['user_input']
    text = str(text.decode(errors='ignore').encode('utf-8', errors='ignore'))

    if VERBOSE:
        print 'predict called on:'
        print text
        print type(text)

    df = pd.Series([text])

    X = PROCESSOR.transform(df)
    resultTO = MODELTO.predict(X)
    resultFROM = MODELFROM.predict(X)

    data = '''
    <h2>Prediction model results:<br> <h2>Sent from: {0}</h2>
    <img src='/Employee.jpg'>

    <br> <h2>Sent to: {1}</h2>
    <img src='/Employee.jpg'></h2><br>
    <form action="/predict" method='POST' >
            <textarea name="user_input" cols="80" rows="15" >{2}</textarea>
            <br><input type="submit" />
    <br><br><br>
    
    '''.format(resultFROM, resultTO, text)

    return render_template('emailRank_home.html').format(data)



## About page

@app.route('/about')
def about():
    data = '''Email Rank is a tool to help better adjust our email to the expectations and norms of the corporate world.
The app uses the public Enron email database of over 100k emails to predict the corporate title of the sender and sendee of any email based on its text. 
Using the prediction and importance scores, the app provides insight into why the email ranks the way is does and how you can improve the way people will perceive you.
    '''
    return render_template('emailRank_home.html').format(data)


## Contact page
@app.route('/contact')
def contact():

    

    data = ''' Created by Alex Bergin as a capstone project for <a href="http://www.galvanize.com/courses/data-science/">Galvanize Data Science.</a><br>
    <img src='/Alex.jpg' height="400" width="400"><br>
    Alex has 5+ years of experience at a premier global business consulting firm. There he developed a reputation for extensive quantitative skills combined with a history of successful client relationships.
He has been recognized as a leader in pricing/contracting analysis having lead teams working on predictive analysis for reimbursement rates and optimization of contract terms.
Alex is focused on continuing to applying his quantitative skills to improve the solutions to complex business problems.
<br>
<a href="https://docs.google.com/document/d/1-VUpr-vOjXB8WmOJQY0qX6n0_ZoWFlZK1k9p6gQqF84/edit?usp=sharing">Resume</a>
<br><span class="email">atbergin (at) gmail (dot) com</span>
    '''


    return render_template('emailRank_home.html').format(data)


## Picture fetching code
@app.route('/Alex.jpg')
def alex():
    with open('data/images/alex.jpg') as pic:
        return pic.read(), 200, {'Content-Type': 'image/jpg'}
@app.route('/Employee.jpg')
def Employee():
    with open('data/images/Employee.jpg') as pic:
        return pic.read(), 200, {'Content-Type': 'image/jpg'}



## Loads the model pickles for use in the app
if __name__ == '__main__':

    print '#################### SERVER START ##############################'

    with open('data/model_pickles/processor.pkl') as f:
        PROCESSOR = pickle.load(f)
        #PROCESSOR = None
    with open('data/model_pickles/modelTO.pkl') as f:
        MODELTO = pickle.load(f)
    with open('data/model_pickles/modelFROM.pkl') as f:
        MODELFROM = pickle.load(f)

    print 'Model Pickles Loaded Successfully'

    # Start Flask app
    app.run(host='0.0.0.0', port=PORT, debug=True)
