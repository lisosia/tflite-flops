import argparse

import tflite_flops

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('input_model', help='Input TFLite model')

    args = parser.parse_args()

    tflite_flops.calc_flops(args.input_model)

if __name__ == '__main__':
    main()
