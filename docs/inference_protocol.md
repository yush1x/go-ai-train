# 推理服务二进制协议

## 启动

默认权重路径：

```text
data/weights/agon_go_net.pt
```

启动服务：

```bash
uv run uvicorn src.server:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

返回：

```json
{"status":"ok"}
```

## POST /predict

请求和响应都使用 raw binary float32，字节序为 little-endian。

### 请求 Header

```http
Content-Type: application/octet-stream
X-Batch-Size: <B>
```

### 请求 Body

Body 是连续的 little-endian float32 数组：

```text
shape  : [B, 9, 19, 19]
layout : batch -> channel -> row -> col
dtype  : float32 little-endian
```

每个输入局面的 9 个通道：

```text
0: 当前局面黑棋
1: 当前局面白棋
2: 前 1 个局面黑棋
3: 前 1 个局面白棋
4: 前 2 个局面黑棋
5: 前 2 个局面白棋
6: 前 3 个局面黑棋
7: 前 3 个局面白棋
8: 当前行动方，黑=1，白=0
```

请求 body 长度：

```text
B * 9 * 19 * 19 * 4 bytes
```

### 响应 Body

响应也是连续的 little-endian float32 数组，按以下顺序拼接：

```text
policy_probs      [B, 362]
value             [B]
score             [B]
ownership_probs   [B, 2, 19, 19]
```

每个 batch 元素对应的 float32 数量：

```text
362 + 1 + 1 + 2 * 19 * 19 = 1086
```

响应 body 长度：

```text
B * 1086 * 4 bytes
```

Go 端读取响应后按以下长度切片：

```text
policy_len    = B * 362
value_len     = B
score_len     = B
ownership_len = B * 2 * 19 * 19
```

切片顺序：

```text
policy_probs    = out[0 : policy_len]
value           = out[policy_len : policy_len + value_len]
score           = out[policy_len + value_len : policy_len + value_len + score_len]
ownership_probs = out[policy_len + value_len + score_len : policy_len + value_len + score_len + ownership_len]
```
