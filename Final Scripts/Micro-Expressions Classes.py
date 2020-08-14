import pandas

dataset_path = "dataset"
micro_expressions = pandas.read_csv(dataset_path + '/Annotation/Microexpressions.csv')
micro_expressions.head()

micro_expressions_merged = micro_expressions.to_string(header=False, index=False, index_names=False).split('\n')
micro_expressions_classes = set()
print("Micro expressions rows =", len(micro_expressions_merged)) #121
for micro_expression in micro_expressions_merged:
    micro_expressions_classes.add(micro_expression[20:139].replace(" ", ""))
print("Unique micro expressions =", len(micro_expressions_classes)) #105

with open('micro_expressions_classes.txt', 'w+') as f:
    for micro_expression_class in micro_expressions_classes:
        f.write("%s\n" % micro_expression_class)

#Creating the labels for deceptive and truthful videos. Label = index of the micro-expression in the numpy array which is written in the file. Then that index converted to be in one-hot encoding.

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

current_directory = os.getcwd()
os.mkdir(current_directory + dataset_path + '/VGG/Labels')
np.save(dataset_path + '/VGG/Labels/deceptive_labels.npy', deceptive_labels_onehot)
np.save(dataset_path + '/VGG/Labels/truthful_labels.npy', truthful_labels_onehot)