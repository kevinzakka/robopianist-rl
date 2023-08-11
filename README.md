# robopianist-rl

Reinforcement learning code for [RoboPianist](https://github.com/google-research/robopianist).

## Installation

Note: Make sure you are using the same conda environment you created for RoboPianist (see [here](https://github.com/google-research/robopianist/blob/main/README.md#installation)).

1. Install [JAX](https://github.com/google/jax#installation)
2. Run `pip install -r requirements.txt`

## Usage

We provide an example bash script to train a policy to play Twinkle Twinkle Little Star with the task parameters used in the paper.

```bash
bash run.sh
```

To look at all the possible command-line flags, run:

```bash
python train.py --help
```

## Citation

If you use this code, please cite our paper:

```bibtex
@article{zakka2023robopianist,
  author = {Zakka, Kevin and Smith, Laura and Gileadi, Nimrod and Howell, Taylor and Peng, Xue Bin and Singh, Sumeet and Tassa, Yuval and Florence, Pete and Zeng, Andy and Abbeel, Pieter},
  title = {{RoboPianist: A Benchmark for High-Dimensional Robot Control}},
  journal = {arXiv preprint arXiv:2304.04150},
  year = {2023},
}
```
