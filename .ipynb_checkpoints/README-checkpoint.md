# Capsule Network MNIST Visualization

This capsule network based on Keras(ref, https://github.com/XifengGuo/CapsNet-Keras) and visualize MNIST using 2D digits capsules!

## Usage

We simply saved hyper parameters in `hyperparams.py` file.
You can use each python script without input parameters.  

#### Model training

If you want to train model, check the `base model` in `models.py` and just start `python3 train.py` or see the `train_with_notebook.ipynb` file.

#### Visualization 2D plot with CapsNet

The `base model`'s final layer's capsules which is called as digits capsules in original paper dimension is 2. So the digits capsules shape is `10 x 2`. More detail, you can see on `visualization-2D-MNIST.ipynb` file.

## References
- Original paper - https://arxiv.org/abs/1710.09829
- CapsNet-Keras - https://github.com/XifengGuo/CapsNet-Keras

