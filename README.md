# RetServe

RetServe 是一个 OpenAI-compatible embedding only 的检索服务。完整链路只有一条：

```text
JSONL corpus -> /v1/embeddings -> .npy embeddings -> FAISS index -> HTTP search API
```

所有编码阶段和查询阶段的 embedding 都走同一个 OpenAI 兼容接口，例如 vLLM、New API、One API 或其他兼容 `/v1/embeddings` 的服务。

## 环境准备

推荐使用项目内虚拟环境，避免污染系统 Python：

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r requirements.txt
```

也可以使用普通 Python 环境：

```bash
python -m pip install -r requirements.txt
```

如果 embedding endpoint 需要鉴权，不要把 key 写进配置文件，使用环境变量：

```bash
export RET_SERVE_EMBED_API_KEY="sk-..."
```

## 输入语料格式

输入语料是 JSONL，每行一个文档。默认读取 `contents` 字段作为待编码文本。

```jsonl
{"id":"1","contents":"刘备\n刘备是三国时期蜀汉昭烈帝。"}
{"id":"2","contents":"诸葛亮\n诸葛亮辅佐刘备，曾有草船借箭等故事。"}
```

服务加载语料时会把 `contents` 第一行作为 `title`，剩余部分作为 `text`。其他字段不会影响编码流程。

## 配置文件

配置都在 `config/` 目录：

- `config/embed.yaml`：离线编码配置，生成 `.npy`
- `config/index.yaml`：FAISS 建索引配置，生成 `.index`
- `config/serve.yaml`：在线检索服务配置
- `config/log.yaml`：日志配置

### `config/embed.yaml`

```yaml
embedding:
  url: "http://58.57.119.12:52010/v1"
  model: "qwen3-embedding-0.6b"
  api_key: "None"
  api_key_env: "RET_SERVE_EMBED_API_KEY"
  batch_size: 128
  concurrency_limit: 16
  encode_batch_size: 4096
  request_timeout: 120.0
  max_retries: 2
  normalize: true

data:
  corpus_path: "./data/san_guo_yan_yi.jsonl"
  embedding_path: "./data/san_guo_yan_yi.npy"
```

参数说明：

| 参数 | 作用 | 建议 |
| --- | --- | --- |
| `embedding.url` | OpenAI-compatible API base URL，通常以 `/v1` 结尾 | 编码和服务必须一致 |
| `embedding.model` | embedding 模型名 | 编码和服务必须一致 |
| `embedding.api_key` | 直接写入的 key | 不推荐写真实 key |
| `embedding.api_key_env` | 从环境变量读取 key | 推荐使用 `RET_SERVE_EMBED_API_KEY` |
| `embedding.batch_size` | 每个 embeddings 请求包含多少条文本 | 从 64/128 开始压测 |
| `embedding.concurrency_limit` | 同时发起多少个 embeddings 请求 | 从 8/16 开始，按服务吞吐调大 |
| `embedding.encode_batch_size` | 每次从 JSONL 缓冲多少条后写 `.npy` | 通常设为 `batch_size * concurrency * 2~8` |
| `embedding.request_timeout` | 单次请求超时时间，秒 | 大 batch 可设 120 或更高 |
| `embedding.max_retries` | OpenAI client 内部重试次数 | 服务稳定时 1~2 |
| `embedding.normalize` | 是否 L2 归一化向量 | 使用 `IndexFlatIP` 做 cosine 时设为 `true` |
| `data.corpus_path` | 输入 JSONL 路径 | 大规模语料建议放高速磁盘 |
| `data.embedding_path` | 输出 `.npy` 路径 | 文件会增量写入 |

### `config/index.yaml`

```yaml
index:
  chunk_size: 50000
  use_gpu: false
  mmap: true
  normalize: false

data:
  embedding_path: "./data/san_guo_yan_yi.npy"
  index_path: "./data/san_guo_yan_yi.index"
```

参数说明：

| 参数 | 作用 | 建议 |
| --- | --- | --- |
| `index.chunk_size` | 每次向 FAISS 添加多少条向量 | 5 万到 20 万之间调 |
| `index.use_gpu` | 建索引阶段是否使用 GPU | 当前保存逻辑以 CPU index 为主，通常 `false` |
| `index.mmap` | 是否 mmap 读取 `.npy` | 千万级语料建议 `true` |
| `index.normalize` | 建索引时是否再归一化 | 如果编码时 `embedding.normalize=true`，这里设 `false` |
| `data.embedding_path` | 输入 `.npy` 路径 | 必须和编码输出一致 |
| `data.index_path` | 输出 FAISS index 路径 | 服务端读取该文件 |

### `config/serve.yaml`

```yaml
server:
  host: "0.0.0.0"
  port: 8088
  max_topk: 999

index:
  path: "./data/san_guo_yan_yi.index"
  use_gpu: true
  gpu_ids: "5"
  search_concurrency_limit: 128

data:
  corpus_path: "./data/san_guo_yan_yi.jsonl"

embedding:
  url: "http://58.57.119.12:52010/v1"
  model: "qwen3-embedding-0.6b"
  api_key: "None"
  api_key_env: "RET_SERVE_EMBED_API_KEY"
  batch_size: 128
  concurrency_limit: 32
  request_timeout: 60.0
  max_retries: 2
  normalize: true
```

参数说明：

| 参数 | 作用 | 建议 |
| --- | --- | --- |
| `server.host` | 服务监听地址 | 对外提供服务用 `0.0.0.0` |
| `server.port` | 服务端口 | 默认 `8088` |
| `server.max_topk` | 限制单次请求最大 topk | 防止误请求拖垮服务 |
| `index.path` | FAISS index 路径 | 必须指向建索引输出 |
| `index.use_gpu` | 查询阶段是否把 FAISS index 放到 GPU | GPU 资源足够时可设 `true` |
| `index.gpu_ids` | GPU ID 列表 | 当前 FAISS 转 GPU 使用可见设备中的第 0 张 |
| `index.search_concurrency_limit` | CPU index 并发 search 数 | CPU 模式可调大，GPU 模式会强制串行 |
| `data.corpus_path` | 原始 JSONL 路径 | 文档顺序必须和编码时一致 |
| `embedding.*` | 查询 embedding 配置 | `url/model/normalize` 必须和编码阶段一致 |

## 全流程运行

### 1. 检查 embedding endpoint

```bash
export RET_SERVE_EMBED_API_KEY="sk-..."

curl http://58.57.119.12:52010/v1/embeddings \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer ${RET_SERVE_EMBED_API_KEY}" \
  -d '{"model":"qwen3-embedding-0.6b","input":["测试 embedding","三国时期刘备的主要事迹"]}'
```

确认返回 `data` 数量和输入数量一致，并且 embedding 维度符合预期。

### 2. 编码语料

```bash
.venv/bin/python embed.py --config embed
```

编码器会：

- 先统计 JSONL 行数
- 按 `encode_batch_size` 流式读取文本
- 按 `batch_size` 切分 API 请求
- 按 `concurrency_limit` 并发请求 embedding 服务
- 将结果直接写入 `.npy.tmp`
- 全部成功后原子替换成最终 `.npy`

如果任务中断，临时文件会清理，避免留下半成品覆盖正式 embedding 文件。

### 3. 构建 FAISS index

```bash
.venv/bin/python index.py --config index
```

默认使用 `IndexFlatIP + IndexIDMap2`。如果 `embedding.normalize=true`，内积检索等价于 cosine similarity。

### 4. 启动服务

```bash
.venv/bin/python ret_serve.py --config serve
```

启动后访问：

- `GET /health`
- `POST /search`
- `GET /docs`

### 5. 请求示例

```bash
curl http://localhost:8088/search \
  -H "Content-Type: application/json" \
  -d '{"queries":["诸葛亮和刘备的关系"],"topk":3}'
```

响应格式：

```json
{
  "contents": [
    [
      {
        "id": "2",
        "title": "诸葛亮",
        "text": "诸葛亮辅佐刘备，曾有草船借箭等故事。",
        "contents": "诸葛亮\n诸葛亮辅佐刘备，曾有草船借箭等故事。"
      }
    ]
  ],
  "scores": [[0.65]]
}
```

## 大规模语料建议

千万级 Wikipedia 语料重点关注这几个参数：

| 阶段 | 参数 | 调优方向 |
| --- | --- | --- |
| 编码 | `embedding.batch_size` | 提高单请求吞吐，直到 API 延迟或显存压力明显上升 |
| 编码 | `embedding.concurrency_limit` | 提高并发，直到 embedding 服务 QPS 或网络带宽达到瓶颈 |
| 编码 | `embedding.encode_batch_size` | 控制内存缓冲，通常 2K 到 32K |
| 编码 | `embedding.request_timeout` | 大 batch 时调高，避免慢请求误超时 |
| 索引 | `index.chunk_size` | 太小会慢，太大占内存 |
| 索引 | `index.mmap` | 大 `.npy` 必须建议开启 |
| 服务 | `index.use_gpu` | index 放得进 GPU 时可显著降低 search 延迟 |
| 服务 | `embedding.concurrency_limit` | 查询阶段一般比离线编码低，避免服务抖动 |

一致性原则：

- 编码和服务的 `embedding.model` 必须一致。
- 编码和服务的 `embedding.normalize` 必须一致。
- 建索引时不要重复归一化，除非 `.npy` 不是归一化后的结果。
- `data.corpus_path` 的文档顺序必须和编码时完全一致，因为 FAISS ID 使用 JSONL 行号。

## 常见问题

### `401 Invalid token`

确认环境变量已设置：

```bash
echo "$RET_SERVE_EMBED_API_KEY"
```

确认 curl 使用的是 Bearer token：

```bash
-H "Authorization: Bearer ${RET_SERVE_EMBED_API_KEY}"
```

### 检索分数不对

优先检查：

- 编码和查询是否使用同一个模型。
- 编码和查询是否都设置了相同的 `normalize`。
- `index.normalize` 是否造成了重复归一化。
- JSONL 是否在编码后被重新排序或改写。

### 内存占用过高

降低：

- `embedding.encode_batch_size`
- `embedding.batch_size`
- `index.chunk_size`

同时确认：

- `index.mmap: true`
- 输入输出路径在本地高速磁盘上

### GPU FAISS 加载失败

服务会回退到 CPU index。检查：

- 是否安装了 GPU 版 FAISS。
- `CUDA_VISIBLE_DEVICES` 和 `index.gpu_ids` 是否正确。
- index 是否过大，无法放进 GPU 显存。

## 项目结构

```text
RetServe/
├── src/
│   ├── embedding_client.py  # OpenAI-compatible embedding client
│   ├── embed.py             # JSONL -> .npy
│   ├── index.py             # .npy -> FAISS index
│   ├── ret_serve.py         # HTTP retrieval service
│   ├── settings.py          # Pydantic config models
│   └── config_loader.py     # YAML loader
├── config/
│   ├── embed.yaml
│   ├── index.yaml
│   ├── serve.yaml
│   └── log.yaml
├── data/
├── embed.py
├── index.py
├── ret_serve.py
└── requirements.txt
```
