[![MseeP.ai Security Assessment Badge](https://mseep.net/pr/shwstone-mcp-oi-wiki-badge.png)](https://mseep.ai/app/shwstone-mcp-oi-wiki)
[![Verified on MseeP](https://mseep.ai/badge.svg)](https://mseep.ai/app/9d61dc89-76bb-401f-840e-07ba2c9cb39b)

# mcp-oi-wiki

让大模型拥有 OI-Wiki 的加成！

![乘法逆元搜索结果](./image.png)

## How does it work?

使用 Deepseek-V3 对 OI-wiki 当前的 462 个页面做摘要，将摘要嵌入为语义向量，建立向量数据库。

查询时，找到数据库中最接近的向量，返回对应的 wiki markdown。

## Usage

确保你拥有 `uv`。

首先，下载本仓库：

```
cd <path of MCP servers>
git clone --recurse-submodules https://github.com/ShwStone/mcp-oi-wiki.git
```

然后打开你的 MCP 配置文件（mcpo 或 claude）：

```json
{
  "mcpServers": {
    "oi-wiki": {
      "command": "uv",
      "args": [
        "--directory",
        "<path of MCP servers>/mcp-oi-wiki",
        "run",
        "python",
        "main.py"
      ]
    }
  }
}
```

## Update

可以生成自己的 `db/oi-wiki.db`。

将 Silicon flow API key 放在 `api.key` 文件中。

然后运行：

```sh
uv run script/request.py
```

在[批量推理页面](https://cloud.siliconflow.cn/batches)下载摘要结果到 `result.jsonl`。

最后运行：

```sh
uv run script/gendb.py
```

生成新的 `db/oi-wiki.db`。

## Thanks

- [milvus-io/milvus-lite: A lightweight version of Milvus](https://github.com/milvus-io/milvus-lite) 向量数据库
- [OI-wiki/OI-wiki: :star2: Wiki of OI / ICPC for everyone. （某大型游戏线上攻略，内含炫酷算术魔法）](https://github.com/OI-wiki/OI-wiki) OI-wiki
- [qdrant/fastembed: Fast, Accurate, Lightweight Python library to make State of the Art Embedding](https://github.com/qdrant/fastembed) CPU 向量嵌入
