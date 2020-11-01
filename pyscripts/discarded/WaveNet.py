
import argparse

parser = argparse.ArgumentParser(description = 'This program allow to train denoise WaveNet from "A Wavenet for Speech Denoising" D. Rethage, J. Pons, X. Serra in a PyTorch fashion provided by https://github.com/Sytronik/denoising-wavenet-pytorch')

#----- Training parameters
parser.add_argument('--min_length', type = int, default = 16000, help = 'The length for the audio signals, in ms.')
parser.add_argument('--batch_size', type = int, default = 8, help = 'The batch size for the training procedure.')

#----- WaveNet parameters
parser.add_argument('--in_channels', type = int, default = 1, help = 'Input channels for WaveNet.')
parser.add_argument('--num_layers', type = int, default = 30, help = 'Number of Conv. Layers Gated Linear Unit.')
parser.add_argument('--num_stacks', type = int, default = 3, help = 'Number of stacks. Must be 10 layers per stack.')
parser.add_argument('--residuals_channels', type = int, default = 128, help = 'Number of channels for residual conv. layers.')
parser.add_argument('--gate_channels', type = int, default = 128, help = 'Number of channels for gate conv. layers.')
parser.add_argument('--skip_out_channels', type = int, default = 128, help = 'Number of channels for skip out conv. layers.')
parser.add_argument('--last_channels', type = tuple, default = (1024, ,256), help = 'Number of channels for the last layers.')

parser.add_argument('--colab', default = False, action = 'store_true')
parser.add_argument('--eval', default = False, action = 'store_true')

args = parser.parse_args()
