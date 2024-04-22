from viewer.viewer import GLFWApp
from core.env import Env
import sys

## Arg parser
import argparse
parser = argparse.ArgumentParser(description='Muscle Simulation')
parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint_path')
parser.add_argument('--env_path', type=str, default='data/env.xml', help='Env_path')
parser.add_argument('--no_vqvae_plot', action='store_true', help='No VQ-VAE plot')

if __name__ == "__main__":
    args = parser.parse_args()
    app = GLFWApp()
    app.draw_vqvae_plot = not args.no_vqvae_plot
    if args.checkpoint:
        app.loadNetwork(args.checkpoint)
    else:
        env_str = None
        with open(args.env_path, "r") as file:
            env_str = file.read()
        app.setEnv(Env(env_str))


    app.startLoop()
