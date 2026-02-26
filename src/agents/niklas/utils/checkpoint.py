import os
from dataclasses import asdict

from framework.common import logger as log


class CheckpointManager:
    def __init__(self, agent, run_name, storage_path):
        self.agent = agent
        self.run_name = run_name
        self.storage_path = storage_path

        if not os.path.exists(self.storage_path):
            os.makedirs(self.storage_path)

    def store_dict(self):
        """Collects state dicts from agent for checkpoint saving"""
        state = {"actor": self.agent.actor.state_dict() if self.agent.actor else None}

        if hasattr(self.agent, "processor") and hasattr(self.agent.processor, "rms"):
            rms = self.agent.processor.rms
            state["obs_rms"] = {"mean": rms.mean, "var": rms.var, "count": rms.count}

        state["config"] = asdict(self.agent.cfg)

        for attr in ["enc", "c1", "c2", "critic", "actor_opt", "critic_opt", "critics"]:
            obj = getattr(self.agent, attr, None)
            if obj is not None:
                state[attr] = obj.state_dict()

        return state

    def load_dict(self, state_dict, inference=False):
        """
        Loads state dicts
        """

        # Backward compatability for time awarness
        # TODO convert old checkpoints
        if "config" in state_dict and hasattr(self.agent, "cfg"):
            saved_time_aware = state_dict["config"].get("time_aware", False)
        elif "enc" in state_dict:
            enc_weights = state_dict["enc"]
            weight_key = next((k for k in enc_weights.keys() if "weight" in k), None)

            if weight_key is not None:
                in_features = enc_weights[weight_key].shape[1]
                saved_time_aware = in_features == 19
                log.info(
                    f"Inferred time_aware={saved_time_aware} from weight shape {in_features}"
                )
            else:
                saved_time_aware = self.agent.cfg.time_aware
        else:
            saved_time_aware = self.agent.cfg.time_aware

        if saved_time_aware != self.agent.cfg.time_aware:
            log.warning(
                f"Architecture Mismatch: Switching time_aware to {saved_time_aware}"
            )

            self.agent.cfg.time_aware = saved_time_aware
            self.agent.time_aware = saved_time_aware

            from agents.niklas.utils.setup import build_networks, setup_dimensions

            setup_dimensions(self.agent)

            build_networks(self.agent)
            log.info("Rebuilt Networks to match inferred checkpoint architecture")

        if "config" in state_dict:
            self.agent.cfg.update(state_dict["config"])

        if "obs_rms" in state_dict and hasattr(self.agent, "processor"):
            rms_data = state_dict["obs_rms"]
            self.agent.processor.rms.mean = rms_data["mean"]
            self.agent.processor.rms.var = rms_data["var"]
            self.agent.processor.rms.count = rms_data["count"]

        if hasattr(self.agent, "enc") and "enc" in state_dict and self.agent.enc:
            self.agent.enc.load_state_dict(state_dict["enc"])

        for attr in ["actor", "critic", "critics", "actor_opt", "critic_opt"]:
            obj = getattr(self.agent, attr, None)
            if obj is not None and attr in state_dict:
                try:
                    obj.load_state_dict(state_dict[attr])
                except Exception as e:
                    print(f"Warning: Failed to load {attr}. Error: {e}")

    def save_snapshot(self, time_step, cfg):
        """
        Creates a lightweight snapshot of the agent for self-play
        """
        snapshot = self.agent.__class__(
            run_name=f"{self.run_name}_snap_{time_step}",
            storage_path=self.storage_path,
            config=self.agent.argument_config,
            total_steps=0,
            inference_mode=True,
            agent_kwargs=cfg.__dict__,
        )

        def copy_net(name):
            if hasattr(self.agent, name) and hasattr(snapshot, name):
                src = getattr(self.agent, name)
                dst = getattr(snapshot, name)
                if src is not None and dst is not None:
                    dst.load_state_dict(src.state_dict())
                    dst.eval()

        copy_net("actor")
        copy_net("enc")

        return snapshot
