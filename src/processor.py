from pathlib import Path
from sgfmill import sgf
from sgfmill import boards
import numpy as np
import h5py

folder = Path("./data/games/Agon") # 需要处理的sgf文件目录
save_path = Path("./data/supervised/Agon.h5")
save_path.parent.mkdir(parents=True, exist_ok=True)
board_size = 19
shard_size = 100000 # 每多少个样本保存一次

state_list = []
policy_list = []
value_list = []

def encode_board(board):
    planes = np.zeros((2, board_size, board_size), dtype=np.int8)
    for r in range(board_size):
        for c in range(board_size):
            stone = board.get(r, c)
            if stone == "b": # 通道0: 黑棋位置
                planes[0, r, c] = 1
            elif stone == "w": # 通道1: 白棋位置
                planes[1, r, c] = 1
    return planes

def create_h5():
    with h5py.File(save_path, "w") as f:
        f.create_dataset("state", shape=(0, 9, board_size, board_size), maxshape=(None, 9, board_size, board_size), dtype=np.int8, chunks=True)
        f.create_dataset("policy", shape=(0,), maxshape=(None,), dtype=np.int64, chunks=True)
        f.create_dataset("value", shape=(0,), maxshape=(None,), dtype=np.float32, chunks=True)

def append_h5():
    if len(state_list) == 0:
        return

    states = np.stack(state_list, axis=0)  # [N, 9, 19, 19], int8
    policies = np.array(policy_list, dtype=np.int64)  # [N]
    values = np.array(value_list, dtype=np.float32)  # [N]

    with h5py.File(save_path, "a") as f:
        old = f["state"].shape[0]
        new = old + len(states)

        f["state"].resize(new, axis=0)
        f["policy"].resize(new, axis=0)
        f["value"].resize(new, axis=0)

        f["state"][old:new] = states
        f["policy"][old:new] = policies
        f["value"][old:new] = values

    print(f"saved {save_path}, state={states.shape}, policy={policies.shape}, value={values.shape}")

    state_list.clear()
    policy_list.clear()
    value_list.clear()


def check(game):
    if game.get_size() != board_size: # 检查棋盘大小
        return False

    return True

def process_game(game):
    # 1. 检查是否棋局是否合法
    if not check(game):
        return

    # 2. 遍历棋局生成多个样本
    board = boards.Board(board_size)  # 默认规则可以用中国规则
    history = []
    for node in game.get_main_sequence():

        move = node.get_move()
        color, pos = move
        if color is None:
            continue

        # 2.1 生成当前棋局状态的 numpy 数组
        current_planes = encode_board(board)
        recent = [current_planes] + history[-3:][::-1]
        state = np.zeros((9, board_size, board_size), dtype=np.int8)
        for i, planes in enumerate(recent):
            state[i * 2] = planes[0]
            state[i * 2 + 1] = planes[1]
        state[8, :, :] = 1 if color == "b" else 0 # 通道8: 当前先手颜色

        # 2.2 模拟当前落子步骤
        if pos is None:
            print("检测到 pass")
            policy = board_size * board_size # pass
        else:
            row, col = pos
            policy = row * board_size + col
            try:
                board.play(row, col, color)
                history.append(encode_board(board))
            except Exception as e:
                print("检测到非法走法，已跳过该棋局")
                return

        # 2.3 生成 label
        if not game.get_winner():
            value = 0
        elif game.get_winner() == color:
            value = 1
        else:
            value = -1

        # 2.4 将样本添加至list
        state_list.append(state)
        policy_list.append(policy)
        value_list.append(value)


def main():
    create_h5()

    for path in folder.rglob("*.sgf"): # 递归遍历目录
        try:
            game = sgf.Sgf_game.from_bytes(path.read_bytes())
            process_game(game)
            if len(state_list) >= shard_size: # 达到阈值则保存
                append_h5()
        except Exception as e:
            print(f"处理失败: {path} | {e}")

    if len(state_list) > 0: # 若最后一批次样本数量足够则保存
        append_h5()

if __name__ == "__main__":
    main()
