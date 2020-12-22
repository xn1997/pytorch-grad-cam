# 自定义修改内容

1. 添加mask和image的融合比例系数
2. target layer不存在时，默认将feature module的最后输出作为feature，避免查询target layer
3. 修改使输入图片不在为固定的正方形

## Grad-CAM implementation in Pytorch ##

### What makes the network think the image label is 'pug, pug-dog' and 'tabby, tabby cat':
![Dog](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/dog.jpg?raw=true) ![Cat](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cat.jpg?raw=true)

### Combining Grad-CAM with Guided Backpropagation for the 'pug, pug-dog' class:
![Combined](https://github.com/jacobgil/pytorch-grad-cam/blob/master/examples/cam_gb_dog.jpg?raw=true)

Gradient class activation maps are a visualization technique for deep learning networks.

See the paper: https://arxiv.org/pdf/1610.02391v1.pdf

The paper authors torch implementation: https://github.com/ramprs/grad-cam

My Keras implementation: https://github.com/jacobgil/keras-grad-cam


----------


This uses VGG19 from torchvision. It will be downloaded when used for the first time.

The code can be modified to work with any model.
However the VGG models in torchvision have features/classifier methods for the convolutional part of the network, and the fully connected part.
This code assumes that the model passed supports these two methods.


----------


Usage: `python gradcam.py --image-path <path_to_image>`

To use with CUDA:
`python gradcam.py --image-path <path_to_image> --use-cuda`
