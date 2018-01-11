from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import hashlib
import io
import os

import numpy as np
import PIL.Image as pil
import tensorflow as tf
from glob import glob

from object_detection.utils import dataset_util
from object_detection.utils import label_map_util_inv

tf.app.flags.DEFINE_string('data_dir', '/media/lijunyi/Seagate JIONG/599/test', 'Location of root directory for the '
                           'data.')
tf.app.flags.DEFINE_string('output_path', '/media/lijunyi/Seagate JIONG/599/', 'Path to which TFRecord files'
                           'will be written. The TFRecord with the training set'
                           'will be located at: <output_path>_train.tfrecord.'
                           'And the TFRecord with the validation set will be'
                           'located at: <output_path>_val.tfrecord')
tf.app.flags.DEFINE_string('label_map_path', './object_detection/data/car_new_label_map.pbtxt',
                           'Path to label map proto.')
tf.app.flags.DEFINE_integer('validation_set_size', '2128', 'Number of images to'
                            'be used as a validation set.')
FLAGS = tf.app.flags.FLAGS


def prepare_example(image_path, annotations, label_map_dict):
  """Converts a dictionary with annotations for an image to tf.Example proto.

  Args:
    image_path: The complete path to image.
    annotations: A dictionary representing the annotation of a single object
      that appears in the image.
    label_map_dict: A map from string label names to integer ids.

  Returns:
    example: The converted tf.Example.
  """
  with tf.gfile.GFile(image_path, 'rb') as fid:
    encoded_jpg = fid.read()
  print(encoded_jpg[1,1,:])
  encoded_jpg_io = io.BytesIO(encoded_jpg)
  image = pil.open(encoded_jpg_io)
  image = np.asarray(image)

  key = hashlib.sha256(encoded_jpg).hexdigest()
  width = int(image.shape[1])
  height = int(image.shape[0])

  xmin_norm = annotations['2d_bbox_left'] / float(width)
  ymin_norm = annotations['2d_bbox_top'] / float(height)
  xmax_norm = annotations['2d_bbox_right'] / float(width)
  ymax_norm = annotations['2d_bbox_bottom'] / float(height)

  example = tf.train.Example(features=tf.train.Features(feature={
      'image/height': dataset_util.int64_feature(height),
      'image/width': dataset_util.int64_feature(width),
      'image/filename': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/source_id': dataset_util.bytes_feature(image_path.encode('utf8')),
      'image/key/sha256': dataset_util.bytes_feature(key.encode('utf8')),
      'image/encoded': dataset_util.bytes_feature(encoded_jpg),
      'image/format': dataset_util.bytes_feature('jpg'.encode('utf8')),
      'image/object/bbox/xmin': dataset_util.float_list_feature(xmin_norm),
      'image/object/bbox/xmax': dataset_util.float_list_feature(xmax_norm),
      'image/object/bbox/ymin': dataset_util.float_list_feature(ymin_norm),
      'image/object/bbox/ymax': dataset_util.float_list_feature(ymax_norm),
      'image/object/class/text': dataset_util.bytes_list_feature(
          [label_map_dict[x].encode('utf8') for x in annotations['type']]),
      'image/object/class/label': dataset_util.int64_list_feature(annotations['type']),
  }))
  return example


def read_annotation_file(filename):
  """Reads a KITTI annotation file.

  Converts a KITTI annotation file into a dictionary containing all the
  relevant information.

  Args:
    filename: the path to the annotataion text file.

  Returns:
    anno: A dictionary with the converted annotation information. See annotation
    README file for details on the different fields.
  """
  try:
      bbox = np.fromfile(filename, dtype=np.uint32)
  except:
      print('[*] bbox not found.')
      bbox = np.array([], dtype=np.uint32)
  bbox.resize([bbox.size // 6, 6])

  anno = {}
  anno['2d_bbox_left'] = np.array(bbox[:,0]-bbox[:,2]/2.0).astype(int)
  anno['2d_bbox_top'] = np.array(bbox[:,1]-bbox[:,3]/2.0).astype(int)
  anno['2d_bbox_right'] = np.array(bbox[:,0]+bbox[:,2]/2.0).astype(int)
  anno['2d_bbox_bottom'] = np.array(bbox[:,1]+bbox[:,3]/2.0).astype(int)
  anno['type'] = bbox[:,4]
  return anno


def convert_self_to_tfrecords(data_dir, output_path,
                               label_map_path, validation_set_size):

  label_map_dict = label_map_util_inv.get_label_map_dict(label_map_path)

  image_files = glob(os.path.join(data_dir,'*/*_image.jpg'))
  annotation_files = glob(os.path.join(data_dir,'*/*_bbox_2d.bin'))

  train_writer = tf.python_io.TFRecordWriter('%s_train.tfrecord'%
                                             output_path)
  val_writer = tf.python_io.TFRecordWriter('%s_val.tfrecord'%
                                           output_path)
  np.random.seed(42)
  image_files_1=image_files[10000:20000]

  for idx in np.random.permutation(len(image_files)):
      print(idx)
      is_validation_img = idx < validation_set_size
      img_anno = read_annotation_file(annotation_files[idx])
      example = prepare_example(image_files[idx], img_anno, label_map_dict)
      if is_validation_img:
          val_writer.write(example.SerializeToString())
      else:
          train_writer.write(example.SerializeToString())

  train_writer.close()
  val_writer.close()

def main(_):
  convert_self_to_tfrecords(
      data_dir=FLAGS.data_dir,
      output_path=FLAGS.output_path,
      label_map_path=FLAGS.label_map_path,
      validation_set_size=FLAGS.validation_set_size)

if __name__ == '__main__':
  tf.app.run()
