# tflite-flops
roughly calculate fops of tflite model.
only Conv and DepthwiseConv layers are considered for now.

### Install
```
pip3 install git+https://github.com/lisosia/tflite-flops
```

### Usage
```
python3 -m tflite_flops example.tflite
```

### exapmle
```
wget https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
tar xvf mobilenet_v2_1.0_224.tgz
python3 -m tflite_flops ./mobilenet_v2_1.0_224.tflite

#=> below printed
OP_NAME            | M FLOPS
------------------------------
CONV_2D            | 21.7
DEPTHWISE_CONV_2D  | 7.2
CONV_2D            | 12.8
.
.
.
CONV_2D            | 2.6
RESHAPE            | <IGNORED>
SOFTMAX            | <IGNORED>
------------------------------
Total: 601.6 M FLOPS
```
