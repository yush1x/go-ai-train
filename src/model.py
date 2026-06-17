import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # 1. Conv3×3 + BatchNorm + ReLU
        # 2. Conv3×3 + BatchNorm
        # 3. output = output + x
        # 4. ReLU
        residual = x
        output = self.bn1(self.conv1(x))
        output = self.relu(output)
        output = self.bn2(self.conv2(output))
        output = output + residual
        output = self.relu(output)
        return output

class GoNet(nn.Module):
    def __init__(self, in_channels, channels, num_blocks, board_size=19):
        super().__init__()

        # 输入层：in_channels -> channels
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )

        # 残差主干
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_blocks)]
        )

        # policy head
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, board_size * board_size + 1), # 19 * 19 + 1 (pass)
        )

        # value head
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 2, 1, bias=False),
            nn.BatchNorm2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2 * board_size * board_size, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.stem(x)  # [B, in_channels, 19, 19] -> [B, channels, 19, 19]
        x = self.res_blocks(x)  # [B, channels, 19, 19] -> [B, channels, 19, 19]

        policy = self.policy_head(x)  # [B, board_size * board_size + 1]
        value = self.value_head(x).squeeze(-1)  # [B]

        return policy, value
