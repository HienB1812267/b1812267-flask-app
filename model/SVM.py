import pandas as pd
from sklearn import svm
from sklearn.model_selection import train_test_split
import pickle

#-----------------------------#

iris = pd.read_csv("../iris.csv")
x = iris.drop(columns=['variety'])
y = iris.variety
#train theo ty le 80/20
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size= 0.2)

#Xay dung mo hinh
model = svm.SVC(kernel='rbf')
model.fit(x_train, y_train)

with open('./model_SVM.pkl','wb') as f:
    pickle.dump(model,f)

predictions = model.predict(x_test)

print("Do chinh xác cua mo hinh voi nghi thuc kiem tra hold-out: %.3f" % 
model.score(x_test, y_test))