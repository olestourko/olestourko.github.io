---
layout: post
title: "Generating ConvNet training data with Blender - Part 1: Automating Renders" 
categories: [machine learning, python, blender]
image: 'assets/posts/generating-convnet-training-data-with-blender-1.jpg'
---

## Overview
I'm working on a machine learning project to detect trash in specific settings, like on grass or the bank of a river.
There doesn't seem to be a large and labelled training dataset available for this, and training ConvNets requires a LOT of data.
As an alternative to taking thousands of photos myself and labelling them, I'm trying to create synthetic data using Blender scripts:

1. Use Blender's physics engine to randomly drop trash into the scene.
2. Render the scene at many different camera angles. Save an image of each render.
3. For each render, determine which objects are visible in the scene and their bounding boxes. Save these labels to a file.

At the end of this part, we'll have a system to automatically create renders and labels for a scene with placeholder objects.

## Blender Scene

I created a Blender scene with two placeholder objects, a cube and sphere. I set them up for rigidbody physics so that they can be dropped in later on.
My Blender setup and scripts are available [here](https://github.com/olestourko/ml-garbage-classifier-tensorflow/tree/master/blender-data-genenerator).

![Render](/assets/posts/blender-render-1.jpg)
![Scene](/assets/posts/blender-scene.jpg)

A couple notes about my scene setup:

- The camera is constrained to an Empty at [0, 0, 0] so that it keeps facing the same point whenever you move it.
- I'm using the hair particle system to render grass, and an environment map for the background.

## The Scripts

Blender uses Python for scripting, and it works _very_ well. Basically anything you can do with the UI, you can also do with a script.

**scene_setup.py**: sets up the scene by randomly positioning the objects and running the physics sim.
**boundingbox.py**: determines the bounding box for the object, if it is visible to the camera.    
**batch_render.py**: has a method for rendering scenes at a bunch of different angles and writing the labels for each to a file.

### scene_setup.py
This script is pretty simple. It randomly positions the objects somewhere above the ground and runs the physics sim for 100 frames.

{% highlight python %}
import bpy
import random

def simulate(scene, mesh_objects, spawn_range, p_visible):
    scene.frame_set(0)
    for object in mesh_objects:
        if random.uniform(0, 1) <= p_visible:
            object.hide = False
            object.hide_render = False
        else:
            object.hide = True
            object.hide_render = True

        object.location.x = random.randrange(spawn_range[0][0], spawn_range[0][1])
        object.location.y = random.randrange(spawn_range[1][0], spawn_range[1][1])
        object.location.z = random.randrange(spawn_range[2][0], spawn_range[2][1])

    for i in range(1, 100):
        scene.frame_set(i)
{% endhighlight %}

### boundingbox.py
Figures out the bounding boxes for an object if its visible, returns `None` otherwise. Check out the comments to understand how it works.

{% highlight python %}
""" https://blender.stackexchange.com/questions/7198/save-the-2d-bounding-box-of-an-object-in-rendered-image-to-a-text-file """

import bpy
import numpy as np

def camera_view_bounds_2d(scene, camera_object, mesh_object):
    """
    Returns camera space bounding box of the mesh object.

    Gets the camera frame bounding box, which by default is returned without any transformations applied.
    Create a new mesh object based on mesh_object and undo any transformations so that it is in the same space as the
    camera frame. Find the min/max vertex coordinates of the mesh visible in the frame, or None if the mesh is not in view.

    :param scene:
    :param camera_object:
    :param mesh_object:
    :return:
    """

    """ Get the inverse transformation matrix. """
    matrix = camera_object.matrix_world.normalized().inverted()
    """ Create a new mesh data block, using the inverse transform matrix to undo any transformations. """
    mesh = mesh_object.to_mesh(scene, True, 'RENDER')
    mesh.transform(mesh_object.matrix_world)
    mesh.transform(matrix)

    """ Get the world coordinates for the camera frame bounding box, before any transformations. """
    frame = [-v for v in camera_object.data.view_frame(scene=scene)[:3]]

    lx = []
    ly = []

    for v in mesh.vertices:
        co_local = v.co
        z = -co_local.z

        if z <= 0.0:
            """ Vertex is behind the camera; ignore it. """
            continue
        else:
            """ Perspective division """
            frame = [(v / (v.z / z)) for v in frame]

        min_x, max_x = frame[1].x, frame[2].x
        min_y, max_y = frame[0].y, frame[1].y

        x = (co_local.x - min_x) / (max_x - min_x)
        y = (co_local.y - min_y) / (max_y - min_y)

        lx.append(x)
        ly.append(y)

    bpy.data.meshes.remove(mesh)

    """ Image is not in view if all the mesh verts were ignored """
    if not lx or not ly:
        return None

    min_x = np.clip(min(lx), 0.0, 1.0)
    min_y = np.clip(min(ly), 0.0, 1.0)
    max_x = np.clip(max(lx), 0.0, 1.0)
    max_y = np.clip(max(ly), 0.0, 1.0)

    """ Image is not in view if both bounding points exist on the same side """
    if min_x == max_x or min_y == max_y:
        return None

    """ Figure out the rendered image size """
    render = scene.render
    fac = render.resolution_percentage * 0.01
    dim_x = render.resolution_x * fac
    dim_y = render.resolution_y * fac

    return (min_x, min_y), (max_x, max_y)
{% endhighlight %}

### batch_render.py
This module has two methods, and is the script I actually run from Blender.

`render()` rotates the camera around its center, rendering each step. It also uses `bounding_box` to figure
out the visibility and bounding boxes in each render, and returns all the labels for the renders.

`batch_render()` combines all the other modules. It sets up a scene with `scene_setup`, renders a bunch of angles with `render()`,
and repeats. It writes all the labels from `render()` to a `labels.json`.

{% highlight python %}
import bpy, os
from math import sin, cos, pi
import numpy as np
import json
import sys
import boundingbox

def render(scene, camera_object, mesh_objects, camera_steps, file_prefix="render"):
    """
    Renders the scene at different camera angles to a file, and returns a list of label data
    """

    radians_in_circle = 2.0 * pi
    original_position = np.matrix([
        [8],
        [0],
        [2]
    ])

    """ This will store the bonding boxes """
    labels = []

    for i in range(0, camera_steps + 1):
        for j in range(0, camera_steps + 1):
            yaw = radians_in_circle * (i / camera_steps)
            pitch = -1.0 * radians_in_circle / 16.0 * (j / camera_steps)
            # Blender uses a Z-up coordinate system instead of the standard Y-up system, therefor:
            # yaw = rotate around z-axis
            # pitch = rotate around y-axis
            yaw_rotation_matrix = np.matrix([
                [cos(yaw), -sin(yaw), 0],
                [sin(yaw), cos(yaw), 0],
                [0, 0, 1]
            ])
            pitch_rotation_matrix = np.matrix([
                [cos(pitch), 0, sin(pitch)],
                [0, 1, 0],
                [-sin(pitch), 0, cos(pitch)]
            ])

            new_position = yaw_rotation_matrix * pitch_rotation_matrix * original_position
            camera_object.location.x = new_position[0][0]
            camera_object.location.y = new_position[1][0]
            camera_object.location.z = new_position[2][0]

            # Rendering
            # https://blender.stackexchange.com/questions/1101/blender-rendering-automation-build-script
            filename = '{}-{}y-{}p.png'.format(str(file_prefix), str(i), str(j))
            bpy.context.scene.render.filepath = os.path.join('./renders/', filename)
            bpy.ops.render.render(write_still=True)

            scene = bpy.data.scenes['Scene']
            label_entry = {
                'image': filename,
                'meshes': {}
            }

            """ Get the bounding box coordinates for each mesh """
            for object in mesh_objects:
                bounding_box = boundingbox.camera_view_bounds_2d(scene, camera_object, object)
                if bounding_box:
                    label_entry['meshes'][object.name] = {
                        'x1': bounding_box[0][0],
                        'y1': bounding_box[0][1],
                        'x2': bounding_box[1][0],
                        'y2': bounding_box[1][1]
                    }

            labels.append(label_entry)

    return labels


def batch_render(scene, camera_object, mesh_objects):
    import scene_setup
    camera_steps = 10
    scene_setup_steps = 10
    spawn_range = [
        (-10, 10),
        (-10, 10),
        (5, 10)
    ]
    labels = []

    for i in range(0, scene_setup_steps):
        scene_setup.simulate(scene, mesh_objects, spawn_range, 0.75)
        scene_labels = render(scene, camera_object, mesh_objects, camera_steps, file_prefix=i)
        labels += scene_labels # Merge lists

    with open('./renders/labels.json', 'w+') as f:
        json.dump(labels, f, sort_keys=True, indent=4, separators=(',', ': '))


if __name__ == '__main__':
    scene = bpy.data.scenes['Scene']
    camera_object = bpy.data.objects['Camera']
    mesh_names = ['Cube', 'Sphere']
    mesh_objects = [bpy.data.objects[name] for name in mesh_names]
    batch_render(scene, camera_object, mesh_objects)
    
{% endhighlight %}

## Results
`batch_render.py` writes rendered images to `./renders/`, as well as a `labels.json` file with the visible objects
and their bounding boxes. This takes a while with the step settings I set in the script above, and produces ~1,000 renders.

Here are some random frames it created:

![Render 1](/assets/posts/0-0y-0p.png)
![Render 2](/assets/posts/0-1y-3p.png)
![Render 3](/assets/posts/0-3y-9p.png)
![Render 4](/assets/posts/4-5y-2p.png)
![Render 5](/assets/posts/0-6y-4p.png)
![Render 6](/assets/posts/7-8y-8p.png)

And here's the `labels.json` entry for one of them (the first image):

{% highlight json %}
    {
        "image": "0-0y-0p.png",
        "meshes": {
            "Cube": {
                "x1": 0.15742501695336883,
                "x2": 0.32712470394223725,
                "y1": 0.37746196244973107,
                "y2": 0.5454313910699651
            },
            "Sphere": {
                "x1": 0.9189575489758851,
                "x2": 1.0,
                "y1": 0.12367120160956185,
                "y2": 0.43482510117790535
            }
        }
    },
{% endhighlight %}


## Notes &amp; Next Steps

This isn't a perfect data source because the images aren't coming from a real-world camera. Having training and test data from different sources
can cause accuracy issues. Still, I'm very happy with my results so far. I think that a potential loss of accuracy is well worth the time that Blender can save me in taking thousands of photos.

Feeding this into my Neural Net is the next step, along with replacing the placeholder objects with proper models of trash.