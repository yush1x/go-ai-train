from pathlib import Path

import numpy as np
import torch

from model import GoNet

# 使用方法：
#   infer = GoInference("./data/weights/agon_go_net.pt")
#   states = np.zeros((1, 9, 19, 19), dtype=np.float32)
#   outputs = infer.predict(states)
#
# predict() 接收 numpy.ndarray，shape 必须是 [B, 9, 19, 19]。
# 输入通常是 0/1 平面，内部会转成 float32。
# 返回值是包含以下 numpy 数组的 dict：
#   policy_probs: [B, 362]
#   value: [B]
#   score: [B]
#   ownership_probs: [B, 2, 19, 19]


class GoInference:
    def __init__(self, weights_path, device=None):
        self.device = self._select_device(device)
        self.model = GoNet(in_channels=9, channels=128, num_blocks=5, board_size=19)

        state_dict = torch.load(Path(weights_path), map_location="cpu")
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()

    def _select_device(self, device):
        if device is not None:
            return torch.device(device)

        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @torch.inference_mode()
    def predict(self, states):
        if not isinstance(states, np.ndarray):
            raise TypeError("states must be a numpy.ndarray")
        if states.shape[1:] != (9, 19, 19):
            raise ValueError(f"states must have shape [B, 9, 19, 19], got {states.shape}")

        states_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)

        policy_logits, value, score, ownership_logits = self.model(states_tensor)
        policy_probs = torch.softmax(policy_logits, dim=1)
        ownership_probs = torch.softmax(ownership_logits, dim=1)

        return {
            "policy_probs": policy_probs.cpu().numpy(),
            "value": value.cpu().numpy(),
            "score": score.cpu().numpy(),
            "ownership_probs": ownership_probs.cpu().numpy(),
        }
