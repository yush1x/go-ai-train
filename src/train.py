from pathlib import Path

from torch import nn
from torch.utils.data import DataLoader

from dataset import TrainDataset
from model import GoNet
import torch

num_epochs = 1
lr = 5e-4
h5_path = Path("./data/supervised/Agon.h5")
weights_path = Path("./data/weights/agon_go_net.pt")
log_interval = 100
value_weight = 1

# 自动选择可用设备
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = GoNet(in_channels=9, channels=128, num_blocks=5, board_size=19).to(device)
criterion_policy = nn.CrossEntropyLoss()
criterion_value = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

dataset = TrainDataset(h5_path)
loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,          # 并行读取数据
    pin_memory=device.type == "cuda", # GPU 训练时加速 CPU -> GPU 拷贝
    persistent_workers=True # 多个 epoch 时复用 worker
)


def train(epoch):
    model.train()
    total_policy_loss = 0
    total_value_loss = 0
    total_loss = 0
    interval_batches = 0
    total_batches = len(loader)

    for batch_idx, (state, policy, value) in enumerate(loader, start=1):
        state = state.to(device)
        policy = policy.to(device)
        value = value.to(device)

        optimizer.zero_grad()
        pred_policy, pred_value, pred_score, pred_ownership = model(state)
        policy_loss = criterion_policy(pred_policy, policy)
        value_loss = criterion_value(pred_value, value)
        loss = policy_loss + value_weight * value_loss
        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_loss += loss.item()
        interval_batches += 1

        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            progress = batch_idx / total_batches * 100
            print(
                f"epoch={epoch} batch={batch_idx}/{total_batches} "
                f"progress={progress:.2f}% "
                f"policy_loss={total_policy_loss / interval_batches:.4f} "
                f"value_loss={total_value_loss / interval_batches:.4f} "
                f"loss={total_loss / interval_batches:.4f}"
            )
            total_policy_loss = 0
            total_value_loss = 0
            total_loss = 0
            interval_batches = 0


def main():
    for epoch in range(num_epochs):
        print(f"epoch={epoch + 1} started")
        train(epoch + 1)

    # 保存模型权重
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"saved model weights to {weights_path}")

if __name__ == "__main__":
    main()
