# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
import os
from dataclasses import asdict

try:
    import wandb
except ModuleNotFoundError:
    raise ModuleNotFoundError("WandB is required to log to Weights and Biases.")

def convert_cfg_to_dict(cfg):
    if hasattr(cfg, "__dict__"):
        return {k: convert_cfg_to_dict(v) for k, v in cfg.__dict__.items() if not k.startswith('__')}
    elif isinstance(cfg, (list, tuple)):
        return [convert_cfg_to_dict(v) for v in cfg]
    elif isinstance(cfg, dict):
        return {k: convert_cfg_to_dict(v) for k, v in cfg.items()}
    else:
        return cfg

class WandbSummaryWriter:
    """WandB-only summary writer. Logs metrics, configs, and models using Weights & Biases."""

    def __init__(self, log_dir: str, cfg):
        self.log_dir = log_dir

        # Get the run name from the folder name
        run_name = os.path.split(log_dir)[-1]

        # Load project and entity from config/env
        try:
            project = cfg["wandb_project"]
        except KeyError:
            raise KeyError("Please specify 'wandb_project' in the runner config.")

        try:
            entity = os.environ["WANDB_USERNAME"]
        except KeyError:
            entity = None

        # Initialize wandb
        wandb.login(key="95f259d16d11e7a90398fe17007cef07212a222b") #fill in your own API key on wandb.ai
        wandb.init(project=project, entity=entity, name=run_name)
        wandb.config.update({"log_dir": log_dir})

        # # Optional tag renaming
        self.name_map = {
            "Train/mean_reward/time": "Train/mean_reward_time",
            "Train/mean_episode_length/time": "Train/mean_episode_length_time",
        }

    def log_dict(self, metrics: dict, step=None):
        """Log a whole dictionary of metrics to WandB."""
        mapped = {self._map_path(k): v for k, v in metrics.items()}
        wandb.log(mapped, step=step)

    def add_scalar(self, tag, scalar_value, global_step=None):
        """Log a scalar value to WandB."""
        if isinstance(scalar_value, float) or isinstance(scalar_value, int):
            wandb.log({self._map_path(tag): scalar_value}, step=global_step)
        else:
            try:
                wandb.log({self._map_path(tag): float(scalar_value)}, step=global_step)
            except Exception:
                print(f"[Warning] Failed to log tag: {tag}, value: {scalar_value}")

    def stop(self):
        """Finish the current WandB run."""
        wandb.finish()

    def log_config(self, env_cfg, runner_cfg, alg_cfg, policy_cfg):
        """Log full config to WandB."""
        wandb.config.update({
            "env_cfg": convert_cfg_to_dict(env_cfg),
            "runner_cfg": convert_cfg_to_dict(runner_cfg),
            "alg_cfg": convert_cfg_to_dict(alg_cfg),
            "policy_cfg": convert_cfg_to_dict(policy_cfg),
        })

    def save_model(self, model_path, iter):
        """Track a model checkpoint."""
        wandb.save(model_path, base_path=os.path.dirname(model_path))

    def save_file(self, path, iter=None):
        """Track an arbitrary file."""
        wandb.save(path, base_path=os.path.dirname(path))

    def _map_path(self, path):
        """Optionally rename paths to cleaner tag names."""
        return self.name_map.get(path, path)
