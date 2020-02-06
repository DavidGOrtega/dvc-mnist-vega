import os
import json
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

import Mnist

dirname = os.path.dirname(__file__)

# LABELS, IMAGES = Mnist.read_images_folder(os.path.join(dirname, '../data/test'))
LABELS, IMAGES = Mnist.read_csv(os.path.join(dirname, '../data/mnist_test.csv'))

META = os.path.join(dirname, '../models/mnist.meta')
MODELS = os.path.join(dirname, '../models/')

init = tf.global_variables_initializer()
with tf.Session() as sess:
    saver = tf.train.import_meta_graph(META)
    saver.restore(sess, tf.train.latest_checkpoint(MODELS))

    graph = tf.get_default_graph()

    x = graph.get_tensor_by_name("x:0")
    y = graph.get_tensor_by_name("y:0")
    softmax = graph.get_tensor_by_name("softmax:0")
    accuracy = graph.get_tensor_by_name("accuracy:0")
    feed_dict = { x: IMAGES, y: LABELS }

    pred = sess.run([softmax, accuracy], feed_dict=feed_dict)
    with open(os.path.join(dirname, '../metrics/eval.json'), 'w') as outfile:
        json.dump({ "accuracy" : str(pred[1]) }, outfile)

    tf_confusion_matrix = tf.confusion_matrix(labels=tf.argmax(LABELS, 1), predictions=tf.argmax(pred[0], 1), num_classes=10)
    tf_confusion_matrix = tf_confusion_matrix.eval()
    
    confusion_matrix = []
    for idx,row in enumerate(tf_confusion_matrix):
        for idy,column in enumerate(row):
            confusion_matrix.append({ "label": "Class " + str(idy), "prediction":  "Class " + str(idx), "count": str(column) })

    with open(os.path.join(dirname, '../metrics/confusion_matrix.json'), 'w') as outfile:
        json.dump(confusion_matrix, outfile)
