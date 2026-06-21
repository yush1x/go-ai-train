from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import SelfPlayDataset
from model import GoNet


num_epochs = 1
lr = 2e-4
selfplay_h5_paths = [
    Path("./data/selfplay/selfplay.h5"),
]
initial_weights_path = Path("./data/weights/agon_go_net.pt")
weights_path = Path("./data/weights/selfplay_go_net.pt")
log_interval = 100

batch_size = 64
value_weight = 1.0
score_weight = 0.1
ownership_weight = 0.2


if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


model = GoNet(in_channels=9, channels=128, num_blocks=5, board_size=19).to(device)
if initial_weights_path.exists():
    model.load_state_dict(torch.load(initial_weights_path, map_location=device))
    print(f"loaded initial model weights from {initial_weights_path}")

criterion_value = nn.MSELoss()
criterion_score = nn.SmoothL1Loss()
criterion_ownership = nn.CrossEntropyLoss(ignore_index=-1)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

if not selfplay_h5_paths:
    raise ValueError("selfplay_h5_paths must contain at least one h5 file")
print("using selfplay h5 files:")
for path in selfplay_h5_paths:
    print(f"  {path}")

dataset = SelfPlayDataset(selfplay_h5_paths)
loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=device.type == "cuda",
    persistent_workers=True,
)


def soft_policy_loss(logits, target_policy):
    target_policy = target_policy / target_policy.sum(dim=1, keepdim=True).clamp_min(1e-8)
    log_probs = torch.log_softmax(logits, dim=1)
    return -(target_policy * log_probs).sum(dim=1).mean()


def train(epoch):
    model.train()
    total_policy_loss = 0
    total_value_loss = 0
    total_score_loss = 0
    total_ownership_loss = 0
    total_loss = 0
    interval_batches = 0
    total_batches = len(loader)

    for batch_idx, (features, policy, value, score, ownership) in enumerate(loader, start=1):
        features = features.to(device)
        policy = policy.to(device)
        value = value.to(device)
        score = score.to(device)
        ownership = ownership.to(device)

        optimizer.zero_grad()
        pred_policy, pred_value, pred_score, pred_ownership = model(features)

        policy_loss = soft_policy_loss(pred_policy, policy)
        value_loss = criterion_value(pred_value, value)
        score_loss = criterion_score(pred_score, score)
        ownership_loss = criterion_ownership(pred_ownership, ownership)
        loss = (
            policy_loss
            + value_weight * value_loss
            + score_weight * score_loss
            + ownership_weight * ownership_loss
        )

        loss.backward()
        optimizer.step()

        total_policy_loss += policy_loss.item()
        total_value_loss += value_loss.item()
        total_score_loss += score_loss.item()
        total_ownership_loss += ownership_loss.item()
        total_loss += loss.item()
        interval_batches += 1

        if batch_idx % log_interval == 0 or batch_idx == total_batches:
            progress = batch_idx / total_batches * 100
            print(
                f"epoch={epoch} batch={batch_idx}/{total_batches} "
                f"progress={progress:.2f}% "
                f"policy_loss={total_policy_loss / interval_batches:.4f} "
                f"value_loss={total_value_loss / interval_batches:.4f} "
                f"score_loss={total_score_loss / interval_batches:.4f} "
                f"ownership_loss={total_ownership_loss / interval_batches:.4f} "
                f"loss={total_loss / interval_batches:.4f}"
            )
            total_policy_loss = 0
            total_value_loss = 0
            total_score_loss = 0
            total_ownership_loss = 0
            total_loss = 0
            interval_batches = 0


def main():
    for epoch in range(num_epochs):
        print(f"epoch={epoch + 1} started")
        train(epoch + 1)

    weights_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), weights_path)
    print(f"saved model weights to {weights_path}")


if __name__ == "__main__":
    main()
