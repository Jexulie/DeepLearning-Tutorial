from lenet import LeNet
from miniVGGNet import MiniVGGNet
from keras.utils import plot_model

#_ (28x28x1) 10 - classes
model = LeNet.build(28, 28, 1, 10)
plot_model(model, to_file="LeNet.png", show_shapes=True)
#_ (32x32x3) 10 - classes
model_2 = MiniVGGNet.build(width=32, height=32, depth=3, classes=10)
plot_model(model_2, to_file="miniVGGNet.png", show_shapes=True)