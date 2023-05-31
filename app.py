from flask import Flask
from src.logger import logging
from src.exception import CustomException
import os, sys

app= Flask(__name__)

@app.route('/', methods= ['GET', 'POST'])
def index():

    try:
        raise Exception("We are testing our custom file")
    except Exception as e:
        abc= CustomException(e, sys)
        logging.info(abc.error_message)
        return "Welcome to Machine Learning Project"


if __name__=="__main__":
    app.run(debug=True)





"""from flask import Flask
from src.logger import logging

app= Flask(__name__)

@app.route('/', methods= ['GET', 'POST'])
def index():
    logging.info("We are testing our second method of logging.")
    return "Welcome to Machine Learning Project"

if __name__=="__main__":
    app.run(debug= True)
"""