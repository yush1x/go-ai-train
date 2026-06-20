## Go AI Train 项目文档

## 环境
本项目使用 uv 进行python环境管理，为了保证在不同设备上的兼容性，其中 pytorch 请通过 uv pip Install 安装（不要使用 uv add），并不会写入 pyproject.toml 中。

## 项目结构

```text
src/processor.py   SGF 转监督训练数据
src/dataset.py     H5 训练数据读取
src/model.py       ResNet 四头模型
src/train.py       监督训练入口
src/train_selfplay.py  自博弈训练入口
src/inference.py   numpy 输入输出的推理封装
src/server.py      FastAPI 二进制推理服务
src/selfplay_storage.py  自博弈训练数据解码与 HDF5 追加
docs/              接口协议文档
```

## 数据集
请将数据集和模型的权重放在 `data/` 下，这个目录目前已经gitignore了，不要放在别的地方导致被上传。

## 文档

```text
docs/overview.md             项目整体概况
docs/inference_protocol.md   推理服务二进制协议
docs/storage.md              自博弈训练数据提交协议
```

## 启动推理服务

推理服务加载的权重路径在 `src/server.py` 的 `WEIGHTS_PATH` 中配置。

```bash
uv run uvicorn src.server:app --host 127.0.0.1 --port 8000
```

健康检查：

```bash
curl http://127.0.0.1:8000/health
```

如果需要让局域网内其他进程访问，可将 host 改为 `0.0.0.0`。
