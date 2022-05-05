import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.svm import SVC, LinearSVC
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_wine
from sklearn.feature_selection import SelectKBest, chi2, RFE
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import (cross_val_score, KFold, cross_validate,
                                     train_test_split)
from sklearn.ensemble import StackingClassifier

data = load_wine()
y = data.target
X = data.data
stc = StandardScaler()
lenc = LabelEncoder()
columns = data.feature_names
df = pd.DataFrame(data=np.hstack(tup=(X, y.reshape(-1, 1))),
                  columns=np.hstack(tup=(columns, ["Class"])))
X_std = stc.fit_transform(df[columns])
pipesvm = Pipeline([("stc", stc), ("selection", RFE(LinearSVC())),
                    ("svm", SVC(kernel="linear"))])
pipelda = Pipeline([("stc", stc), ("svm", LinearDiscriminantAnalysis())])
estimators = [("LDA", pipelda), ("SVM", pipesvm)]
# El utilizar clasificadores apilados tiene beneficios cuando se trata de
# problemas multiclase, puesto que puede mejorar mucho el pronostico de clase
# al explotar el poder predictivo del pronostico para ciertas clases
stacking_classifier = StackingClassifier(estimators=estimators,
                                         final_estimator=GaussianNB())
print("Stacking stimators")
print(cross_val_score(X=df[columns], y=y, estimator=stacking_classifier,
                      cv=KFold(5)))
print("Only SVM")
print(cross_val_score(X=df[columns], y=y, estimator=pipesvm, cv=KFold(5)))
print("Only LDA")
print(cross_val_score(X=df[columns], y=y, estimator=pipelda, cv=KFold(5)))

X_train, X_test, y_train, y_test = train_test_split(df[columns], y,\
    test_size=0.33, random_state=42)
stacking_classifier.fit(X_train, y_train)
y_pred = stacking_classifier.predict(X_test)
print("\n")
print(confusion_matrix(y_test, y_pred))
print("\n")
print(classification_report(y_test, y_pred))
