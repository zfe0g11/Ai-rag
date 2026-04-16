from abc import ABC, abstractmethod
from typing import List, Dict, Tuple, Any
from openai import OpenAI
import config
from text2sql.knowledge_base import Text2SQLKnowledgeBase
from text2sql.sql_generator import SQLGenerator
from text2sql.sql_executor import SQLExecutor

class RAGChain(ABC):
    """RAG链抽象基类"""
    
    @abstractmethod
    def retrieve(self, query: str, limit: int) -> List[Dict]:
        """检索相关文档"""
        pass
    
    @abstractmethod
    def generate(self, query: str, documents: List[Dict]) -> str:
        """生成回答"""
        pass
    
    @abstractmethod
    def run(self, query: str) -> str:
        """完整RAG流程"""
        pass


class BasicRAGChain(RAGChain):
    """基础RAG链实现"""
    
    def __init__(self, vector_store, embedding_model, llm_model, api_key, api_base):
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.llm_model = llm_model
        self.client = OpenAI(api_key=api_key, base_url=api_base)
        self.knowledge_base = Text2SQLKnowledgeBase()
        self.sql_generator = SQLGenerator()
        self.sql_executor = SQLExecutor()
        self.max_retry_count = 3
        self.max_result_rows = 100
    
    def retrieve(self, query: str, limit: int) -> List[Dict]:
        """检索相关文档（两层检索）"""
        # 生成查询向量
        from langchain_openai import OpenAIEmbeddings
        
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_LLM_MODEL,
            openai_api_key=config.EMBEDDING_XUNFEI_API_KEY,
            openai_api_base=config.EMBEDDING_XUNFEI_API_BASE
        )
        
        query_vector = embeddings.embed_query(query)
        
        # 第一层检索：在摘要集合中搜索
        print("开始第一层检索：在摘要集合中搜索...")
        summary_results = self.vector_store.search(
            collection_name=config.QDRANT_SUMMARY_COLLECTION_NAME,
            query_vector=query_vector,
            limit=1  # 只需要最相关的摘要
        )
        
        if not summary_results:
            print("未找到相关文档摘要")
            return []
        
        # 获取摘要中的内容集合名称
        summary_doc = summary_results[0]
        collection_name = summary_doc['payload'].get('collection_name', config.QDRANT_COLLECTION_NAME)
        print(f"找到相关文档摘要，内容集合：{collection_name}")
        
        # 第二层检索：在内容集合中搜索
        print("开始第二层检索：在内容集合中搜索...")
        content_results = self.vector_store.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        return content_results
    def sql_generate(self, query: str, result: str) -> str:
        """生成SQL语句"""
        prompt = f"基于以下上下文回答问题：\n\n{result}\n\n问题：{query}\n回答："
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "你是一个助手，根据提供的上下文回答问题。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    
    def generate(self, query: str, documents: List[Dict]) -> str:
        """生成回答"""
        # 构建提示
        context = ""
        for doc in documents:
            context += f"{doc['payload']['content']}\n"
        
        prompt = f"基于以下上下文回答问题：\n\n{context}\n\n问题：{query}\n回答："
        
        # 调用大模型生成回答
        response = self.client.chat.completions.create(
            model=self.llm_model,
            messages=[
                {"role": "system", "content": "你是一个助手，根据提供的上下文回答问题。"},
                {"role": "user", "content": prompt}
            ]
        )
        
        return response.choices[0].message.content
    
    def run(self, query: str, use_database: bool = False) -> str:
        """完整RAG流程"""
        if use_database:
            # 使用数据库查询
            result = self.run_database_query(query)
            answer = self.sql_generate(query, result)
        else:
            # 使用文档查询
            # 检索相关文档
            documents = self.retrieve(query, limit=3)
            
            # 生成回答
            answer = self.generate(query, documents)
            
        return answer
    
    def run_database_query(self, query: str) -> str:
        """运行数据库查询"""
        # 1. 检索相关知识
        print("检索相关知识...")
        knowledge_results = self.knowledge_base.search(query, top_k=5)
        print(f"检索到 {len(knowledge_results)} 条相关信息")

        # 2. 生成SQL
        print("生成SQL...")
        sql = self.sql_generator.generate_sql(query, knowledge_results)
        print(f"生成SQL语句: {sql}")

        # 3. 执行SQL（带重试）
        retry_count = 0
        while retry_count < self.max_retry_count:
            print(f"执行SQL (尝试 {retry_count + 1}/{self.max_retry_count})...")
            
            success, result = self._execute_sql(sql)
            
            if success:
                print("SQL执行成功!")
                return str(result)
            else:
                print(f"SQL执行失败: {result}")
                
                if retry_count < self.max_retry_count - 1:
                    print("尝试修复SQL...")
                    sql = self.sql_generator.fix_sql(sql, result, knowledge_results)
                    print(f"修复后的SQL: {sql}")
                
                retry_count += 1
        
        return f"执行SQL失败：超过最大重试次数 ({self.max_retry_count})"
    
    def _execute_sql(self, sql: str) -> Tuple[bool, Any]:
        """执行SQL语句"""
        try:       
            if sql.strip().upper().startswith('SELECT') and 'LIMIT' not in sql.upper():
                sql = f"{sql.rstrip(';')} LIMIT {self.max_result_rows}"
            # 执行SQL
            results = self.sql_executor.execute(sql)
            
            # 格式化结果
            if results:
                columns = list(results[0].keys())
                rows = [list(row.values()) for row in results]
                return True, {
                    "columns": columns,
                    "rows": rows,
                    "count": len(results)
                }
            else:
                return True, "SQL执行成功"
        
        except Exception as e:
            return False, str(e)