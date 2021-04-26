import pandas
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
import cv2
import matplotlib.pyplot as plt
import os

# ======================================================================================================================
#    get dataset from csv file using pandas
# ======================================================================================================================

train_file = 'b_depressed.csv'
dataset = pandas.read_csv(train_file)

# ======================================================================================================================
#   clasification_column(last one) if someone is depressed or not
#   rest columns 1-22 are classifiers
# ======================================================================================================================

clasification_column = dataset[dataset.columns[-1]]
feature_set = dataset[dataset.columns[1:21]]

# ======================================================================================================================
#   split data set to specific subsets
# ======================================================================================================================

X_train, X_test, Y_train, Y_test = train_test_split(feature_set, clasification_column, test_size=0.33)

# ======================================================================================================================
#  Build a decision tree classifier
# ======================================================================================================================

classifier = DecisionTreeClassifier(max_depth=5)

# ======================================================================================================================
#    Build a decision tree classifier from the training set (XTrain and Y_train) and
#    extract different types of target set be depressed or not (1,0)
#    extract features names
# ======================================================================================================================

classifier.fit(X_train, Y_train)
class_names = clasification_column.unique()
class_names = class_names.astype(str)
feature_names = dataset.columns[1:21]

# ======================================================================================================================
#   generates a GraphViz representation of the decision tree
# ======================================================================================================================

export_graphviz(classifier, 'classifier.dot', feature_names=list(feature_set), class_names=class_names)
os.system('dot  -Tpng classifier.dot -o classifier.png')
img = cv2.imread('classifier.png')
plt.figure(figsize=(13, 13))
ax = plt.gca()
ax.axes.get_xaxis().set_visible(False)
ax.axes.get_yaxis().set_visible(False)
ax.imshow(img)
plt.show()


# ======================================================================================================================
#   predicted class for each sample
# ======================================================================================================================

print(classifier.predict(X_test[0:-1]))

classifier = RandomForestClassifier(max_depth=3, n_estimators=9)
classifier.fit(X_train, Y_train)


print(len(classifier.estimators_))

# ======================================================================================================================
#   create an image for every separate tree
# ======================================================================================================================

for tree_id, tree in enumerate(classifier.estimators_):
    export_graphviz(tree, f'tree{tree_id:02d}.dot', feature_names=feature_names, class_names=class_names)
    os.system(f'dot -Tpng tree{tree_id:02d}.dot -o tree{tree_id:02d}.png')

fig = plt.figure(figsize=(18, 18))
fig.tight_layout(pad=0.8)

for tree_id, tree in enumerate(classifier.estimators_):
    ax = fig.add_subplot(3, 3, tree_id + 1 )
    ax.title.set_text(f'tree{tree_id:02d}')
    img = cv2.imread(f'tree{tree_id:02d}.png')
    ax.imshow(img)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)

plt.show()
