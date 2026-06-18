# Python 推理服务二进制协议

本文档面向 Go 调用方。协议只定义 `POST /predict` 的请求和响应格式。

## 基本约定

```text
URL          : http://<host>:<port>/predict
Content-Type : application/octet-stream
数字格式      : float32
字节序        : little-endian
输入布局      : batch -> channel -> row -> col
```

所有数组都是连续的一维 float32 字节流。Go 端按 little-endian 写入和读取。

## 请求

Header：

```http
Content-Type: application/octet-stream
X-Batch-Size: <B>
```

Body 表示一个 batch 的局面：

```text
shape        : [B, 9, 19, 19]
float32 数量 : B * 9 * 19 * 19
byte 数量    : B * 9 * 19 * 19 * 4
```

每个局面的 9 个通道：

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

Go 端需要按以下索引写入：

```text
idx = (((batch * 9 + channel) * 19 + row) * 19 + col)
```

## 响应

响应 Body 是连续的 float32 字节流，按以下顺序拼接：

```text
policy_probs      [B, 362]
value             [B]
score             [B]
ownership_probs   [B, 2, 19, 19]
```

每个 batch 元素返回：

```text
362 + 1 + 1 + 2 * 19 * 19 = 1086 个 float32
```

响应总长度：

```text
float32 数量 : B * 1086
byte 数量    : B * 1086 * 4
```

Go 端读取成 `[]float32` 后按以下长度切片：

```text
policyLen    = B * 362
valueLen     = B
scoreLen     = B
ownershipLen = B * 2 * 19 * 19
```

切片顺序：

```text
policy    = out[0 : policyLen]
value     = out[policyLen : policyLen + valueLen]
score     = out[policyLen + valueLen : policyLen + valueLen + scoreLen]
ownership = out[policyLen + valueLen + scoreLen : policyLen + valueLen + scoreLen + ownershipLen]
```

## Go 示例

请求：

```go
states := make([]float32, batchSize*9*19*19)

// 示例：写入某个点
idx := (((b*9 + c) * 19 + row) * 19 + col)
states[idx] = 1

buf := new(bytes.Buffer)
if err := binary.Write(buf, binary.LittleEndian, states); err != nil {
    return err
}

req, err := http.NewRequest("POST", url, buf)
if err != nil {
    return err
}
req.Header.Set("Content-Type", "application/octet-stream")
req.Header.Set("X-Batch-Size", strconv.Itoa(batchSize))

resp, err := http.DefaultClient.Do(req)
if err != nil {
    return err
}
defer resp.Body.Close()
```

响应：

```go
if resp.StatusCode != http.StatusOK {
    body, _ := io.ReadAll(resp.Body)
    return fmt.Errorf("predict failed: status=%d body=%s", resp.StatusCode, body)
}

body, err := io.ReadAll(resp.Body)
if err != nil {
    return err
}

expectedBytes := batchSize * 1086 * 4
if len(body) != expectedBytes {
    return fmt.Errorf("invalid response size: got=%d expected=%d", len(body), expectedBytes)
}

out := make([]float32, len(body)/4)
if err := binary.Read(bytes.NewReader(body), binary.LittleEndian, out); err != nil {
    return err
}

policyLen := batchSize * 362
valueLen := batchSize
scoreLen := batchSize
ownershipLen := batchSize * 2 * 19 * 19

policy := out[:policyLen]
value := out[policyLen : policyLen+valueLen]
score := out[policyLen+valueLen : policyLen+valueLen+scoreLen]
ownership := out[policyLen+valueLen+scoreLen : policyLen+valueLen+scoreLen+ownershipLen]
```

## 错误

服务端会在以下情况返回 `400`：

```text
X-Batch-Size 缺失或小于等于 0
请求 body 长度不是 B * 9 * 19 * 19 * 4
```
