"""Run a pretrained RL policy in the RoboPianist environment.

1. Make sure you checkout commit `75a5d478997693979d801d45b78490edcd4e218a` of RoboPianist.
2. Download the pretrained policy checkpoints from the following link:
    `https://drive.google.com/file/d/1VUFpnq0HoMJbyleRYAA2WcIjyKnL2-pM/view?usp=sharing`
    The file is roughly 3GB.
3. Update the path to the downloaded checkpoints in `BASE_DIR`.
4. Update the song name in `ckpt_dir` to the song of choice (corresponds to a subdir in `BASE_DIR`).
5. Run with `python run_policy.py`.
"""

from pathlib import Path
import yaml
from flax.training import checkpoints

import sac
import specs

import dm_env
import numpy as np
from robopianist import suite, viewer
import dm_env_wrappers as wrappers

# TODO: Replace with path to downloaded checkpoints.
BASE_DIR = Path("/Users/kevin/datasets/full_action_space_piano_policies/")
SEED = 42


def main() -> None:
    ckpt_dir = (
        BASE_DIR
        # TODO: Replace with song of choice.
        / "SAC-piano_RoboPianist-debug-NocturneRousseau-v0-0-1685733688.7507768"
    )
    assert ckpt_dir.exists()

    cfg_filename = ckpt_dir / "hydra" / "config.yaml"
    assert cfg_filename.exists()
    with open(cfg_filename, "r") as f:
        cfg = yaml.load(f, Loader=yaml.FullLoader)

    # ============================== #
    # Environment.
    # ============================== #
    domain_task = cfg["args"]["domain_task"]
    env_name = domain_task.split("_")[1]

    env = suite.load(
        environment_name=env_name,
        seed=SEED,
        task_kwargs=dict(
            n_steps_lookahead=cfg["args"]["n_steps_lookahead"],
            trim_silence=True,
            gravity_compensation=True,
            reduced_action_space=False,
            control_timestep=0.05,
            disable_colorization=False,
            primitive_fingertip_collisions=True,
            change_color_on_activation=True,
        ),
    )

    env = wrappers.ObservationActionRewardWrapper(env)
    env = wrappers.ConcatObservationWrapper(env)
    env = wrappers.CanonicalSpecWrapper(env, clip=True)
    env = wrappers.SinglePrecisionWrapper(env)
    env = wrappers.DmControlWrapper(env)

    # ============================== #
    # Agent.
    # ============================== #
    spec = specs.EnvironmentSpec.make(env)

    sac_cfg = sac.SACConfig(
        activation=cfg["agent_config"]["activation"],
        critic_dropout_rate=cfg["agent_config"]["critic_dropout_rate"],
        critic_layer_norm=cfg["agent_config"]["critic_layer_norm"],
        hidden_dims=cfg["agent_config"]["hidden_dims"],
    )

    agent = sac.SAC.initialize(
        spec=spec,
        config=sac_cfg,
        seed=SEED,
        discount=cfg["args"]["discount"],
    )

    # Restore checkpoint.
    agent = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=agent)

    def policy(timestep: dm_env.TimeStep) -> np.ndarray:
        return agent.eval_actions(timestep.observation)

    viewer.launch(env, policy=policy)


if __name__ == "__main__":
    main()
