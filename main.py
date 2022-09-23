import cv2
import tensorflow as tf
import numpy as np
import os

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util

camera = cv2.VideoCapture(0)

label_file_path = os.path.join('C:/Users/edward.bromilow/PycharmProjects/detectorProject', 'my_retrained_map.pbtxt')

model_file_path = 'C:/Users/edward.bromilow/PycharmProjects/detectorProject/my_retrained_graph.pb'

object_types = 101

retrained_graph = tf.Graph()

with retrained_graph.as_default():

    od_graph_def = tf.GraphDef()

    with tf.gfile.GFile(model_file_path, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name ='')


label_map = label_map_util.load_labelmap(label_file_path)
categories = label_map_util.convert_label_map_to_categories(
    label_map, max_num_classes=object_types, use_display_name=True)

category_index = label_map_util.create_category_index(categories)


def load_ims(image):
    (w, h) = image.size
    return np.array(image.getdata()).reshape((w, h, 3)).astype(np.uint8)


with retrained_graph.as_default():
    with tf.Session(graph=retrained_graph) as sess:
        while True:
            ret, image_np = camera.read()
            expanded_image = np.expand_dims(image_np, axis=0)
            object_classes = retrained_graph.get_tensor_by_name('detection_classes:0')

            number_of_detections = retrained_graph.get_tensor_by_name('num_detections:0')

            image_tensor = retrained_graph.get_tensor_by_name('image_tensor:0')

            container = retrained_graph.get_tensor_by_name('detection_boxes:0')

            scores = retrained_graph.get_tensor_by_name('detection_scores:0')

            (boxes, scores, classes, num_detections) = sess.run(
                [container, scores, object_classes, number_of_detections],
                feed_dict={image_tensor: expanded_image})

            vis_util.visualize_boxes_and_labels_on_image_array(
                image_np, np.squeeze(boxes), np.squeeze(classes).astype(np.int32),
                np.squeeze(scores), category_index, use_normalized_coordinates=True,
                line_thickness=4)

            cv2.imshow('Detecting', cv2.resize(image_np, (1200, 720)))

            if cv2.waitKey(25) & 0xFF == ord('e'):
                cv2.destroyAllWindows()
                break