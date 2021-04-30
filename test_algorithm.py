import Pre_processing_the_data as pp
from sklearn.svm import SVC
from sklearn import metrics

""" The classification is done with the help of SVC from the scikit-learn
and the accuracy of about 90% is obtained """
X_train_pca=pp.pca.transform(pp.X_train)
X_test_pca=pp.pca.transform(pp.X_test)
clf = SVC()
clf.fit(X_train_pca, pp.y_train)
y_pred = clf.predict(X_test_pca)
print("accuracy score:{:.2f}".format(metrics.accuracy_score(pp.y_test, y_pred)))


"""The confusion matrix is calculated with the help of metrics lib """

cm=metrics.confusion_matrix(pp.y_test,y_pred)
print(cm)


"""there was a function in scikit-learn to display the complete classification report 
but then the issue did not require the complete classification report 
thus only the accuracy of our classification is displayed """

# print("Classification Results:\n{}".format(metrics.classification_report(pp.y_test, y_pred)))