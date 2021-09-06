import pandas as pd
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import pickle

#-----------------------------#

iris = pd.read_csv("../iris.csv")
x = iris.drop(columns=['variety'])
y = iris.variety
#train theo ty le 80/20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

#xay dung mo hinh voi giai thuat cay quyet dinh
model = tree.DecisionTreeClassifier(criterion="gini")
model.fit(x_train, y_train)

predictions = model.predict(x_test)

with open('./model_iris_decision_tree.pkl','wb') as f:
    pickle.dump(model,f)

#do chinh xac
print("Do chinh xac cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" % accuracy_score(y_test, predictions))
