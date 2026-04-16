# AI-rag 项目

## 项目简介
AI-rag 是一个基于检索增强生成（RAG）的项目，支持文档处理、向量存储和Text2SQL功能。

## 目录结构

```
AI-rag/
├── config.py              # 配置文件
├── main.py                # 主程序
├── document_processor.py  # 文档处理器
├── vector_store.py        # 向量存储
├── rag_chain.py           # RAG链
├── text2sql/              # Text2SQL模块
│   ├── __init__.py
│   ├── knowledge_base.py  # 知识库
│   ├── sql_generator.py   # SQL生成器
│   ├── sql_executor.py    # SQL执行器
│   └── data/              # 知识库数据
│       ├── sql_examples.json
│       └── table_schemas.json
└── text2sql_demo.py       # Text2SQL演示程序
```

## 功能模块

### 1. 文档处理
- 支持多种文档格式（docx、pdf、md等）
- 文档分割和向量化
- 向量存储和检索

### 2. 向量存储
- 基于Qdrant的向量数据库
- 支持两层检索（摘要索引和内容索引）
- 文档摘要和内容分离存储

### 3. Text2SQL功能
- 自然语言查询生成SQL语句
- 知识库存储SQL示例和表结构
- 支持多种数据库类型（sqlite、postgresql、mysql）

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 文档处理和向量存储

```python
import config
from document_processor import UniversalDocumentProcessor
from vector_store import QdrantVectorStore
from rag_chain import BasicRAGChain

# 初始化文档处理器
doc_processor = UniversalDocumentProcessor()

# 初始化向量存储
vector_store = QdrantVectorStore()
vector_store.connect(config.QDRANT_URL)

# 初始化RAG链
rag_chain = BasicRAGChain(
    vector_store=vector_store,
    embedding_model=config.EMBEDDING_LLM_MODEL,
    llm_model=config.LLM_MODEL,
    api_key=config.XUNFEI_API_KEY,
    api_base=config.XUNFEI_API_BASE
)

# 处理文档
split_docs = doc_processor.process_document(
    file_path=config.DOCUMENT_PATH,
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP
)

# 添加文档到向量存储
# ...

# 运行查询
answer = rag_chain.run("请介绍一下知识库中的内容")
print(answer)
```

### 2. Text2SQL功能

```python
from text2sql import Text2SQLKnowledgeBase, SQLGenerator, SQLExecutor

# 初始化知识库
knowledge_base = Text2SQLKnowledgeBase()
knowledge_base.connect()
knowledge_base.load_data()

# 初始化SQL生成器
sql_generator = SQLGenerator(knowledge_base)

# 初始化SQL执行器
db_config = {
    "type": "sqlite",
    "database": "demo.db"
}
sql_executor = SQLExecutor(db_config)

# 生成SQL
query = "查询所有用户的姓名和邮箱"
sql = sql_generator.generate_sql(query)
print(f"生成的SQL: {sql}")

# 执行SQL
results = sql_executor.execute(sql)
print(f"查询结果: {len(results)} 条记录")
```

## 配置文件

配置文件 `config.py` 包含以下配置：

- Qdrant数据库配置
- 文档处理配置
- 嵌入模型配置
- LLM模型配置

## 演示程序

运行 `text2sql_demo.py` 可以演示Text2SQL功能：

```bash
python text2sql_demo.py
```

## 注意事项

1. 确保Qdrant数据库服务已启动
2. 配置文件中的API密钥和URL需要根据实际情况修改
3. 文档处理和向量存储需要一定的时间和资源
4. Text2SQL功能需要先加载知识库数据

## 扩展功能

可以根据需要扩展以下功能：

- 支持更多的文档格式
- 优化文档分割算法
- 支持更多的数据库类型
- 增强SQL生成的准确性
- 添加知识库自动学习功能

## 许可证

本项目采用MIT许可证。
