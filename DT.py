import csv
import pandas
import pydotplus
from sklearn import tree
from sklearn import preprocessing
from sklearn.feature_extraction import DictVectorizer

# fp = open('E:/python/20171119/t.xlsx','r')
reader = pandas.read_excel('E:/python/20171119/t.xlsx')
header = reader.columns.values
# print(reader)
# print(header)

featuresList = []
labelList = []


labelList = reader['class_language'].values
# print(reader.values)
for i in range(0,len(reader.index)):
    rowdict = {}
    for j in range(1,len(header)-1):
        rowdict[header[j]] = reader.ix[i][j]
        # print(reader.ix[i][j])
        # print(rowdict)
    featuresList.append(rowdict)
# # print(labelList)
print(featuresList)
#
vec = DictVectorizer()
x = vec.fit_transform(featuresList).toarray()
lb = preprocessing.LabelBinarizer()
y = lb.fit_transform(labelList)

print("dunmpyX "+'\n'+str(x) )
print("dunmpyY "+ str(y) )
print("feature_name"+str(vec.get_feature_names()))

clf = tree.DecisionTreeClassifier(criterion='entropy')
clf.fit(x,y)

dot_data = tree.export_graphviz(clf,feature_names=vec.get_feature_names(),out_file=None)
graphziv = pydotplus.graph_from_dot_data(dot_data)
graphziv.write_pdf('dt.pdf')

predict_data = x[0].copy()
predict_data[3] = 1
predict_data[4] = 0

predict_label = clf.predict(predict_data.reshape(1,7))
print(predict_label)
