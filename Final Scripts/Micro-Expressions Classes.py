import pandas
import numpy as np
import os
import tensorflow as tf

current_dir = os.getcwd()
dataset_path = current_dir + "/Dataset"
micro_expressions = pandas.read_csv(dataset_path + '/Annotation/Microexpressions.csv')
micro_expressions.head()

micro_expressions_merged = micro_expressions.to_string(header=False, index=False, index_names=False).split('\n')
micro_expressions_classes = set()
print("Micro expressions rows =", len(micro_expressions_merged)) #121
for micro_expression in micro_expressions_merged:
    micro_expressions_classes.add(micro_expression[20:139].replace(" ", ""))
unique = len(micro_expressions_classes)
print("Unique micro expressions =", unique) #105

micro_expressions_classes = sorted(micro_expressions_classes)
micro_expressions_classes_np = np.asarray(micro_expressions_classes)
with open('micro_expressions_classes.txt', 'w') as f:
    f.write(np.array2string(micro_expressions_classes_np))

# Creating the labels for deceptive and truthful videos. Label = index of the micro-expression in the numpy array which is written in the file. Then that index converted to be in one-hot encoding.

deceptive_labels = []
truthful_labels = []
for row in micro_expressions_merged:
    micro_string = row[20:139].replace(" ", "")
    if row[9] == 'l':  # deceptive
        deceptive_labels.append(np.where(micro_expressions_classes_np == micro_string)[0][0])
    else:
        truthful_labels.append(np.where(micro_expressions_classes_np == micro_string)[0][0])


deceptive_labels_onehot = tf.keras.utils.to_categorical(deceptive_labels, num_classes=unique)
truthful_labels_onehot = tf.keras.utils.to_categorical(truthful_labels, num_classes=unique)

labels_path = current_dir + '/Dataset_VGG/Labels'
os.mkdir(labels_path)
np.save(labels_path + '/deceptive_labels.npy', deceptive_labels_onehot)
np.save(labels_path + '/truthful_labels.npy', truthful_labels_onehot)