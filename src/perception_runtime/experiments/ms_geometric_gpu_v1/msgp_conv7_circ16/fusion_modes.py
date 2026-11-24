"""Select one of the two frozen MSGP-Conv7-Circ16 fusion contracts."""

from agent import MSGPAgent
from dynamic_v22 import MODE, DynamicAgent, DynamicBuilder, attach_frame_observer
from perception import ObservationBuilder


def make_agent(config, fusion_mode, device):
    if fusion_mode == "static":
        return MSGPAgent(config, True, device)
    if fusion_mode == MODE:
        return DynamicAgent(config, True, device)
    raise ValueError(f"Unknown fusion mode: {fusion_mode}")


def make_builder(env, fusion_mode, device):
    if fusion_mode == "static":
        return ObservationBuilder(device, env.lidar.config, True)
    if fusion_mode == MODE:
        attach_frame_observer(env)
        return DynamicBuilder(device, env.lidar.config, True)
    raise ValueError(f"Unknown fusion mode: {fusion_mode}")
