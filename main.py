from flask import Flask,render_template,jsonify,request
# import numpy as np
# import pandas as pd
# import seaborn as sb
# import matplotlib.pyplot as plt
# import sklearn
# from pandas import Series, DataFrame
# # from pylab import rcParams
# from sklearn import preprocessing
# from sklearn.linear_model import LogisticRegression
# from sklearn.model_selection import train_test_split,GridSearchCV
# from sklearn import metrics
# from sklearn.metrics import classification_report
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.preprocessing import StandardScaler
# from sklearn.preprocessing import LabelEncoder,OneHotEncoder
# # from sklearn.externals import joblib
import pickle

app = Flask(__name__)

# url = "https://raw.githubusercontent.com/BigDataGal/Python-for-Data-Science/master/titanic-train.csv"
# df = pd.read_csv(url)
# dt_model2 = []
# df['Age'].fillna(df['Age'].mean(),inplace = True)
#
# x = df.drop(['Survived','PassengerId','Embarked','Name','Cabin','Ticket'],axis =1)
#
# y = df['Survived']
#
# y.value_counts()
#
# lbe_enc = LabelEncoder()
#
# x['label_enco'] = lbe_enc.fit_transform(df['Sex'])
#
# x.drop(['Sex'],axis =1,inplace=True)
#
# train_x,test_x,train_y,test_y = train_test_split(x,y,test_size = .20,random_state =30)
#
# dec_tree = DecisionTreeClassifier()
#
# dec_tree.fit(train_x,train_y)
#
# dec_tree.score(train_x,train_y)
#
# dec_tree.score(test_x,test_y)
#
# path  =  dec_tree.cost_complexity_pruning_path(train_x,train_y)
# ccp_alpha = path.ccp_alphas
#
# for ccp in ccp_alpha:
#     dt_m = DecisionTreeClassifier(ccp_alpha=ccp)
#     dt_m.fit(train_x,train_y)
#     dt_model2.append(dt_m)
# train_score = [i.score(train_x,train_y) for i in dt_model2]
# test_score  = [i.score(test_x,test_y) for i in dt_model2]
# fix,ax = plt.subplots()
# ax.set_xlabel("apha")
# ax.set_ylabel("accuracy")
# ax.plot(ccp_alpha,train_score,marker = 'o',label = 'train')
# ax.plot(ccp_alpha,test_score,marker = 'o',label = 'test')
# ax.legend()
#
# plt.savefig('Titanicgraph.png')
#
# new_model = DecisionTreeClassifier(ccp_alpha=0.009)
#
# new_model.fit(train_x,train_y)
# new_model.score(train_x,train_y)
# new_model.score(test_x,test_y)
#
# param = {
#            'criterion' : ['gini','entropy'], ## One is the gini impurity and another one is entropy low and Information gain is high
#            'splitter' : ['best'], ## when this parameter is fixed to best it will calculate the criterion on each feature and and give the best feature for splitting nodes and if it is RANDOM it will use random features fro splitting
#            'max_depth' : range(10),# specify upto which the tree branch to be splitted if it is default none it will go upto every tree pure split
#            'min_samples_leaf':range(1,5),
#            'min_samples_split':range(1,10)
#              }
#
# grid = GridSearchCV(dec_tree,param_grid=param,cv=10,n_jobs=2,verbose=3)
#
# grid.fit(train_x,train_y)
#
# # grid_pa = grid.best_params_
#
# new_model_tuning = DecisionTreeClassifier(criterion = 'entropy',
#  max_depth = 3,
#  min_samples_leaf = 1,
#  min_samples_split = 2,
#  splitter= 'best',
#   ccp_alpha=0.009)
#
# new_model_tuning.fit(train_x,train_y)
#
# new_model_tuning.score(train_x,train_y)
#
# new_model_tuning.score(test_x,test_y)
# print("ACCURACYYYYYYYYYYYYYYYYYYYYYYYYYYYY",new_model_tuning.score(train_x,train_y),new_model_tuning.score(test_x,test_y))
#
# model_load = pickle.dump(new_model_tuning,open('titanic_survivied_data','wb'))

@app.route('/', methods = ['POST','GET'])
def home():
    return render_template('index.html')




@app.route('/redirect', methods=['POST'])  # This will be called from UI
def math_operation():
    result = ""
    if (request.method=='POST'):
        try:
            Pclass = float(request.form['Pclass'])
            Age = float(request.form['Pclass'])
            SibSp = float(request.form['SibSp'])
            Parch = float(request.form['Parch'])
            Fare  = float(request.form['Fare'])
            if request.form['Sex'] == 'Male':
                Sex =1
            else:
                Sex = 0
            # Sex = request.form['Sex']

            with open('titanic_survivied_data.pkl','rb') as f:
                model = pickle.load(f)
                prediction = model.predict([[Pclass,Age,SibSp,Parch,Fare,Sex]])
            if prediction[0] == 1:
               result = "The Predicted Output is {0} and the passenger survived".format(prediction[0])
            else:
                result = "The Predicted Output is {0} and the passenger is has not survived".format(prediction[0])
        except Exception as e:
            result = "Something Went Wrong"
        return render_template('results.html',result=result)
    else:
        return render_template('index.html')

if __name__ == "__main__":
    app.run(debug=True)
