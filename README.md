# tflite-flops
Roughly calculate FLOPs (floating-point operations) of a tflite format model.  

### Install
```
pip3 install git+https://github.com/lisosia/tflite-flops
```

### Usage
```
python3 -m tflite_flops example.tflite
```

### Exapmle
```
wget https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz
tar xvf mobilenet_v2_1.0_224.tgz
python3 -m tflite_flops ./mobilenet_v2_1.0_224.tflite
```
below lines printed
```
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

### How is it calculated?

In the case of Conv layer
```
Multiply-Accumulate (MAC) = output_h * output_w * output_c * kernel_h * kernel_w * input_c 
                         (= output_h * output_w * weight_size)
Floating-point operations (FLOPs) = 2 * MAC
```

### Limitation

Only Conv and DepthwiseConv layers are considered for now. It is enough for most of the time.
