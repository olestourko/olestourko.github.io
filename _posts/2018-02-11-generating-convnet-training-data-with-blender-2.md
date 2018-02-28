---
layout: post
title: "Generating ConvNet training data with Blender - Part 2: Feeding a Neural Net" 
categories: [machine learning, python, blender]
image: 'assets/posts/generating-convnet-training-data-with-blender-1.jpg'
---

## Overview
In this post, I'll be feeding the renders and labels generated in [Part 1]({{'2018/02/03/generating-convnet-training-data-with-blender-1.html' | absolute_url}})
into a neural network. This is a continuation of my attempt at building a system to identify litter. 
At the end of this part, we'll have a way of training a neural network with our generated data.

## Unpacking images and labels
We need a way of making the renders and labels from `labels.json` available to a neural network model. 
The NN model needs to have a `get_batch(batch_size)` function available for grabbing a list of image/label samples.

### Using the `pickle` module
Reading thousands of images into memory and parsing a large JSON file can take a lot of time, and I'd rather not do that
every time we run our script. That's where Python's [pickle](https://www.pythoncentral.io/how-to-pickle-unpickle-tutorial/) module comes in.

In other languages, _picking_ is known as _serialization_ or _marshalling_, but its basically the same thing. It means taking
a data structure or object and converting into a binary representation that can be dumped to a file on disk. That file can then
be used to restore the dumped data at a later point in time.

**Dumping**
{% highlight python %}
with open(pickle_pathname, 'wb+') as fp:
    pickle.dump(self.data, fp)
{% endhighlight %}

**Loading**
{% highlight python %}
with open(pathname, 'rb') as fp:
    data = pickle.load(fp)
{% endhighlight %}

Restoring data from a pickle'd file is very efficient, and perfect in our use case since the images and labels won't change
in between runs of our neural network traning script.

### data_sources.py
This script reads our images and labels, and provides the `get_batch()` function that the NN will need. This script should be
pretty straightforward, and I won't explain every part since its probably just easier to read it.

Note the `_get_visibility_matrices()` function, though. This is used to get a nice visibility matrix for each image:

$\begin{bmatrix}object\\_one\\_visible \\\ object\\_two\\_visible\end{bmatrix} = \begin{bmatrix}0 \\\ 1\end{bmatrix}$

{% highlight python %}
class ImageDataSource(DataSource):
    def __init__(self, pathname, cached=False):
        import os
        import json
        import pickle
        import numpy
        from PIL import Image

        dir_name = os.path.dirname(pathname)
        pickle_pathname = os.path.join(dir_name, 'labels.pickle')
        if cached and os.path.isfile(pickle_pathname):
            with open(pickle_pathname, 'rb') as fp:
                self.data = pickle.load(fp)

        else:
            with open(pathname) as fp:
                data = json.load(fp)
                fp.close()

            self.data = []
            visibility_matrices = self.__get_visibility_matrices(data)

            for i, entry in enumerate(data):
                image_path = os.path.abspath(os.path.join(dir_name, entry['image']))
                image = Image.open(image_path).convert('RGB')
                """
                Fast PIL > numpy conversion
                https://stackoverflow.com/questions/13550376/pil-image-to-array-numpy-array-to-array-python/42036542#42036542
                """
                image_array = numpy.fromstring(image.tobytes(), dtype=numpy.uint8)
                image_array = numpy.reshape(image_array, (image.size[0], image.size[1], 3))
                self.data.append({
                    'image': image_array,
                    'visibility': visibility_matrices[i],
                    'bounding_boxes': [value for key, value in entry['meshes'].items()]
                })

            with open(pickle_pathname, 'wb+') as fp:
                pickle.dump(self.data, fp)

    def get_image_shape(self):
        return self.data[0]['image'].shape[0], self.data[0]['image'].shape[1], 3

    def get_batch(self, batch_size=10):
        assert batch_size <= len(self.data)
        import random
        shuffled_data = self.data[:]
        random.shuffle(shuffled_data)
        batch = shuffled_data[0:batch_size]
        return batch

    def __get_visibility_matrices(self, data):
        """
        :param data: The contents of the label file
        :type data: dict
        :return: A list of one-hot vectors
        """
        import numpy

        """ Figure out how many unique names there are """
        names = set()
        for entry in data:
            for mesh in entry['meshes']:
                names.add(mesh)

        n_unique_names = len(names)

        matrices = []
        for entry in data:
            matrix = numpy.zeros(shape=(n_unique_names))
            for i, name in enumerate(entry['meshes']):
                matrix[i] = 1.

            matrices.append(matrix)

        return matrices
{% endhighlight %}        
        
## Using the data in a Neural Network

Since this post is about _feeding_ a neural network model, I won't go in-depth about building a good model. I'm using the
sample [CIFAR10 model from the TensorFlow docs](https://www.tensorflow.org/tutorials/deep_cnn) with some extra tweaks. This
neural network has 2 convolutional+pooling layers and 3 fully connected layers.

Note that I'm **ignoring the bounding box labels**, at least for now.

If you just want to see I feed the data, look at the `train()` function.

### Tweak: Using sigmoid instead of softmax

We have a classification problem where multiple objects can be visible at the scene at one time.  
This means that the sum of the label features can be > 1. [Softmax activation](https://www.depends-on-the-definition.com/guide-to-multi-label-classification-with-neural-networks/)
is not appropriate because softmax tries to normalize all the features such that the sum == 1. This is great for multi-label
problems where you want to know the _probability_ of each label, but it doesn't work for our label format.

For example, a correct visibility prediction of
$\begin{bmatrix}1 \\\ 1\end{bmatrix}$
would be transformed into
$\begin{bmatrix}0.5 \\\ 0.5\end{bmatrix}$
by softmax.
 
Instead of the softmax activation function from the NN model in Tensorflow docs, I'm using sigmoid.
In practicality, this means using `tf.nn.sigmoid_cross_entropy_with_logits(...)`
instead of `tf.nn.softmax_cross_entropy_with_logits(..)`.

### nn.py
{% highlight python %} 
import tensorflow as tf
from data_sources import ImageDataSource

"""
Refer to https://www.tensorflow.org/tutorials/deep_cnn
"""

BATCH_SIZE=32
NUM_LABELS=2
ds = ImageDataSource(pathname='../blender-data-generator/renders/labels.json', cached=True)
IMAGE_SHAPE = ds.get_image_shape()

def inference(images):
    """ Conv1 """
    with tf.variable_scope('conv1') as scope:
        kernel = tf.get_variable('weights', [5, 5, 3, 64], initializer=None, dtype=tf.float32)
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv = tf.nn.conv2d(images, kernel, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv1 = tf.nn.relu(pre_activation, name=scope.name)

    """ Pool1
        Takes and returns a tensor in NHWC format: https://www.tensorflow.org/api_docs/python/tf/nn/max_pool
    """
    pool1 = tf.nn.max_pool(conv1, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool1')
    """ Norm 1
        Do this to prevent saturation: https://www.tensorflow.org/api_guides/python/nn#Normalization
    """
    norm1 = tf.nn.local_response_normalization(pool1, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm1')

    """ Conv2 """
    with tf.variable_scope('conv2') as scope:
        kernel = tf.get_variable('weights', [5, 5, 64, 64], initializer=None, dtype=tf.float32)
        biases = tf.get_variable('biases', [64], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        conv = tf.nn.conv2d(norm1, kernel, strides=[1, 1, 1, 1], padding='SAME')
        pre_activation = tf.nn.bias_add(conv, biases)
        conv2 = tf.nn.relu(pre_activation, name=scope.name)

    """ Norm 2 """
    norm2 = tf.nn.local_response_normalization(conv2, depth_radius=4, bias=1.0, alpha=0.001 / 9.0, beta=0.75, name='norm2')
    """ Pool 2 """
    pool2 = tf.nn.max_pool(norm2, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME', name='pool2')

    """ FC1 """
    with tf.variable_scope('fc1') as scope:
        reshape = tf.reshape(pool2, [BATCH_SIZE, -1])
        weights = tf.get_variable('weights', [IMAGE_SHAPE[0]/4*IMAGE_SHAPE[1]/4*64, 384], initializer=None, dtype=tf.float32)
        biases = tf.get_variable('biases', [384], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        fc1 = tf.nn.relu(tf.matmul(reshape, weights) + biases, name=scope.name)

    """ FC2 """
    with tf.variable_scope('fc2') as scope:
        weights = tf.get_variable('weights', [384, 192], initializer=None, dtype=tf.float32)
        biases = tf.get_variable('biases', [192], initializer=tf.constant_initializer(0.1), dtype=tf.float32)
        fc2 = tf.nn.relu(tf.matmul(fc1, weights) + biases, name=scope.name)

    """ Linear """
    with tf.variable_scope('linear') as scope:
        weights = tf.get_variable('weights', [192, NUM_LABELS], initializer=None, dtype=tf.float32)
        biases = tf.get_variable('biases', [NUM_LABELS], initializer=tf.constant_initializer(0.0), dtype=tf.float32)
        linear = tf.add(tf.matmul(fc2, weights), biases, name=scope.name)

    return linear

def loss(logits, labels):
    labels = tf.cast(labels, tf.int32)
    # cross_entropy = tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits, name='cross_entropy_per_example')
    labels = tf.cast(labels, tf.float32) # Need to do this because labels is tf.int32 while logits is tf.float32
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=logits, name='sigmoid_cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')

    return cross_entropy_mean

def train():
    images = tf.placeholder(tf.float32, shape=[None, IMAGE_SHAPE[0], IMAGE_SHAPE[1], 3])
    labels = tf.placeholder(tf.float32, shape=[None, NUM_LABELS])
    with tf.Session() as sess:
        loss_op = loss(inference(images), labels)
        train_step = tf.train.AdamOptimizer(1e-4).minimize(loss_op)

        sess.run(tf.global_variables_initializer())
        for step in range(10000):
            batch = ds.get_batch(BATCH_SIZE)
            batch_images = [item['image'] for item in batch]
            batch_labels = [item['visibility'] for item in batch]

            if step % 100 == 0:
                print 'Loss at step {}: {}'.format(step, loss_op.eval(
                    feed_dict={
                        images: batch_images,
                        labels: batch_labels,
                    }
                ))

            train_step.run(feed_dict={
                images: batch_images,
                labels: batch_labels,
            })

if __name__ == '__main__':
    train()
    
{% endhighlight %}

Running `nn.py` for 10,000 iterations and 32 batch size on a 1060GTX GPU, I got:

{% highlight bash %}
Loss at step 0: 4.66430473328
Loss at step 100: 0.178410932422
Loss at step 200: 0.23581546545
Loss at step 300: 0.316873967648
Loss at step 400: 0.125240594149
Loss at step 500: 0.0854227989912
...
...
...
Loss at step 9900: 4.11001310567e-05
{% endhighlight %}

It seems to work!

## Notes &amp; Next Steps

This neural network is not at all optimized for the data, and I need a better way of measuring accuracy. In the upcoming parts,
I'll measure the performance of the neural network, try to improve it, and also try using the bounding box labels.
I'll also replace the placeholder objects in the scene renders with models of trash.