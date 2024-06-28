from sklearn import svm, datasets, metrics
from sklearn.model_selection import train_test_split

cancer_dataset = datasets.load_breast_cancer()
x_train, x_test, y_train, y_test = train_test_split(cancer_dataset.data, cancer_dataset.target, test_size=0.3, train_size=0.7, random_state=109)

model = svm.SVC(kernel='linear')
model.fit(x_train, y_train)
y_predict = model.predict(x_test)

print("Accuracy:", metrics.accuracy_score(y_test, y_predict))