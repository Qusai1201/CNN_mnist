from tensorflow import keras
import tensorflow as tf
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support , classification_report
from sklearn.metrics import roc_curve, auc

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.astype("float32") / 255
x_test = x_test.astype("float32") / 255


x_train = np.expand_dims(x_train, -1)
x_test = np.expand_dims(x_test, -1)

model = tf.keras.models.load_model('CNN_mnist.h5')

y_pred=model.predict(x_test) 

y_pred_leable = np.argmax(y_pred, axis=1)

precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred_leable, average='weighted')
classification_report = classification_report(y_test , y_pred_leable)
print(classification_report)
print("accuracy_score : " , accuracy_score(y_test , y_pred_leable))
print("Precision: {:.4f}".format(precision))
print("Recall: {:.4f}".format(recall))
print("F1 Score: {:.4f}".format(f1))


Conf_mat = confusion_matrix(y_pred_leable , y_test)


plt.figure(figsize=(10,10))
sns.heatmap(Conf_mat, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()


y_test_one_hot = tf.keras.utils.to_categorical(y_test, num_classes=10)

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(10):
    fpr[i], tpr[i], _ = roc_curve(y_test_one_hot[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

plt.figure(figsize=(10, 10))
colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

for i in range(10):
    plt.plot(fpr[i], tpr[i], color=colors[i], lw=2, label='Digit %d (AUC = %0.2f)' % (i, roc_auc[i]))

plt.plot([0, 1], [0, 1], color='black', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC)')
plt.legend(loc="lower right")
plt.show()
