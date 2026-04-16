# Qdrant配置
QDRANT_URL = "http://172.23.216.45:6333"
QDRANT_COLLECTION_NAME = "rag_knowledge_base"  # 第二层：文档内容集合
QDRANT_SUMMARY_COLLECTION_NAME = "rag_document_summaries"  # 第一层：文档摘要集合
SQL_Xxk_yibang_COLLECTION_NAME = "SQL_Xxk_yibang_databases"  # Text2SQL知识库集合
MYSQL_HOST_URL = "172.23.216.43:3306"
MYSQL_USER = "root"
MYSQL_PASSWORD = "admintoor"
MYSQL_DATABASE = "yibang"

# 文档处理配置
DOCUMENT_PATH_DIR = r"D:\AI开发\Ai-rag\databases"

Delimiter_based_chunking_size = 200
Delimiter_based_chunk_overlap = 40

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200

# 嵌入模型配置
#EMBEDDING_MODEL = "Qwen3-Embedding-0.6B"
LLM_MODEL = "xop3qwen1b7"
XUNFEI_API_BASE = "http://maas-api.cn-huabei-1.xf-yun.com/v1"
XUNFEI_API_KEY = "sk-9UvKsntNsE3WwMWEFc29000bF05e4379AdDd042eB1D76fC2"
# LLM配置
EMBEDDING_LLM_MODEL = "xop3qwen0b6embedding"
EMBEDDING_XUNFEI_API_BASE = "https://maas-api.cn-huabei-1.xf-yun.com/v2"
# 讯飞星辰大模型配置
EMBEDDING_XUNFEI_API_KEY = "a2390186e39f732ec7151112818adbd2:YTJlYzFkNmEwN2M3YzI0ZmFhYmYxNTJi"
