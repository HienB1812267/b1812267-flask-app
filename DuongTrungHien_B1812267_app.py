from flask import Flask, render_template, request, url_for
from flask_cors import CORS, cross_origin
from scipy.stats.stats import mode
from sklearn.naive_bayes import GaussianNB
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
import pandas as pd
import numpy as np

#Init Flask backend server
app = Flask(__name__)

CORS(app)
app.config['CORS_HEADERS'] = "Content-Type"

def bayes_method(data):
    iris = pd.read_csv("./iris.csv")
    x = iris.drop(columns=['variety'])
    y = iris.variety
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)
    model = GaussianNB()
    model.fit(x_train, y_train)

    return model.predict(data)[0]

def decision_tree_method(data):
    iris = pd.read_csv("./iris.csv")
    x = iris.drop(columns=['variety'])
    y = iris.variety
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)
    model = tree.DecisionTreeClassifier(criterion="gini")
    model.fit(x_train, y_train)

    return model.predict(data)[0]

def knn_method(data):
    iris = pd.read_csv("./iris.csv")
    x = iris.drop(columns=['variety'])
    y = iris.variety
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

    model = KNeighborsClassifier(n_neighbors=3)
    model.fit(x_train, y_train)

    return model.predict(data)[0]

def svm_method(data):
    iris = pd.read_csv("./iris.csv")
    x = iris.drop(columns=['variety'])
    y = iris.variety
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)
    model = svm.SVC(kernel='rbf')
    model.fit(x_train, y_train)

    return model.predict(data)[0]


@app.route('/')
@cross_origin(origin='*')
def index_process():
    return render_template("index.html")


@app.route('/predict', methods=["POST", "GET"])
@cross_origin(origin='*')
def predict_process():
    sepalLength = request.form["sepalLength"]
    sepalWidth = request.form["sepalWidth"]
    petalLength = request.form["petalLength"]
    petalWidth = request.form["petalWidth"]

    sample_data = [sepalLength, sepalWidth, petalLength, petalWidth]

    data = [float(i) for i in sample_data]

    final_data = [np.array(data)]

    result_bayes = bayes_method(final_data)
    result_dt = decision_tree_method(final_data)
    result_knn = knn_method(final_data)
    result_svm = svm_method(final_data)

    return render_template("predict.html", result_dt = result_dt, result_bayes=result_bayes, result_knn=result_knn, result_svm=result_svm)

#Start backend server
if __name__ == '__main__':
    app.run(debug=True)