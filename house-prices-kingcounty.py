import pandas
import numpy as np
from sklearn.ensemble import ExtraTreesRegressor
from sklearn import model_selection
#from sklearn import grid_search
######################################################################################

g = pandas.read_csv("kc-house-data.csv",encoding = "ISO-8859-1")
g["price"]    = g["price"]/1000


X               = g[["sqft_above","sqft_basement","sqft_lot","sqft_living","floors","bedrooms",
                     "yr_built","lat","long","bathrooms"]].values
Y               = g["price"].values
zipcodes        = pandas.get_dummies(g["zipcode"]).values
condition       = pandas.get_dummies(g["condition"]).values
grade           = pandas.get_dummies(g["grade"]).values
X               = np.concatenate((X,zipcodes),axis=1)
X               = np.concatenate((X,condition),axis=1)
X               = np.concatenate((X,grade),axis=1)


 
clf            = ExtraTreesRegressor()
parameters     = {'max_depth':np.arange(1,10)}
clfgrid        = model_selection.GridSearchCV(clf, parameters)
clfgrid.fit(X, g["price"].values)
print(clfgrid.grid_scores_)
print(clfgrid.best_params_)

scores = model_selection.cross_val_score(clf,X , g["price"].values, cv=3)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))

g["predicted"] = clfgrid.predict(X)
g["mae"]       = abs(g["predicted"]-g["price"])
g["mae"].mean()
print(g[["predicted","price","mae"]])

#######################################################################################
