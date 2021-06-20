from setuptools import setup

setup(
    name='tflite-flops',
    version='0.0.0',
    description='roughly calculate FLOPS of tflite format model',
    install_requires=['tflite'],
    packages=['tflite_flops'],
    python_requires='>=3.5'
)
