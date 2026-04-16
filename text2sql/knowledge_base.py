import os
import json
import numpy as np
from typing import List, Dict, Any
from langchain_openai import OpenAIEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct
import config


class Text2SQLKnowledgeBase:
    """Text2SQL知识库，用于存储和检索SQL示例和表结构"""
    
    def __init__(self, qdrant_url: str = config.QDRANT_URL):
        self.qdrant_url = qdrant_url
        self.collection_name = config.SQL_Xxk_yibang_COLLECTION_NAME
        self.client = None
        
        # 使用与main.py相同的嵌入模型配置
        self.embedding_function = OpenAIEmbeddings(
            model=config.EMBEDDING_LLM_MODEL,
            openai_api_key=config.EMBEDDING_XUNFEI_API_KEY,
            openai_api_base=config.EMBEDDING_XUNFEI_API_BASE
        )
        
        self.sql_examples = []
        self.table_schemas = []
        self.data_loaded = False
    
    def create_collection(self):
        """创建Qdrant集合"""
        if not self.client:
            raise ValueError("Qdrant客户端未连接，请先设置client属性")
        
        if self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        
        # 嵌入模型维度为768
        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=768, distance=Distance.COSINE)
        )
        
        return True    
    def load_data(self):
        """加载知识库数据"""
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        
        self.load_sql_examples(data_dir)
        self.load_table_schemas(data_dir)
        #self.vectorize_and_store()
        
        self.data_loaded = True    
    def load_sql_examples(self, data_dir: str):
        """加载SQL示例"""
        sql_examples_path = os.path.join(data_dir, "sql_examples.json")
        
        default_examples = [
            {"question": "查询所有汽车的名称和价格", "sql": "SELECT name, price FROM car", "database": "mysql"},
            {"question": "查找价格大于10万的汽车", "sql": "SELECT * FROM car WHERE price > 100000", "database": "mysql"},
            {"question": "查询每个分类的汽车数量", "sql": "SELECT c.name, COUNT(car.id) as car_count FROM category c LEFT JOIN car ON c.id = car.category_id GROUP BY c.id, c.name", "database": "mysql"}
        ]
        
        if os.path.exists(sql_examples_path):
            with open(sql_examples_path, 'r', encoding='utf-8') as f:
                self.sql_examples = json.load(f)
        else:
            self.sql_examples = default_examples
            os.makedirs(data_dir, exist_ok=True)
            with open(sql_examples_path, 'w', encoding='utf-8') as f:
                json.dump(self.sql_examples, f, ensure_ascii=False, indent=2)
        
        # 逐条添加SQL示例
        for example in self.sql_examples:
            self._add_sql_example(example)
    
    def _add_sql_example(self, example: Dict[str, Any]):
        """添加单条SQL示例"""
        content = f"问题: {example['question']}\n"
        content += f"SQL: {example['sql']}\n"
        content += f"数据库: {example.get('database', '')}\n"
        
        self._insert_data([content], ["example"])
    
    def _insert_data(self, contents: List[str], types: List[str]):
        """向量化并插入数据"""
        # 批量向量化
        embeddings = self.embedding_function.embed_documents(contents)
        
        points = []
        for i, (embedding, content, type_) in enumerate(zip(embeddings, contents, types)):
            point = PointStruct(
                id=i+1,
                vector=embedding,
                payload={
                    "content": content,
                    "type": type_
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )
    
    def load_table_schemas(self, data_dir: str):
        """加载表结构信息"""
        schema_path = os.path.join(data_dir, "table_schemas.json")
        
        default_schemas = [
            {
                "table_name": "car",
                "description": "汽车信息表",
                "columns": [
                    {"name": "id", "type": "bigint", "description": "汽车ID，主键，自动递增"},
                    {"name": "name", "type": "varchar(64)", "description": "汽车名称，唯一"},
                    {"name": "brand_id", "type": "bigint", "description": "品牌ID，外键"},
                    {"name": "model_id", "type": "bigint", "description": "型号ID，外键"},
                    {"name": "price", "type": "decimal(10,2)", "description": "汽车价格"},
                    {"name": "image", "type": "varchar(255)", "description": "汽车图片URL"},
                    {"name": "description", "type": "varchar(255)", "description": "汽车描述"},
                    {"name": "status", "type": "int", "description": "状态，默认1"},
                    {"name": "create_time", "type": "datetime", "description": "创建时间"},
                    {"name": "update_time", "type": "datetime", "description": "更新时间"},
                    {"name": "create_user", "type": "bigint", "description": "创建用户ID"},
                    {"name": "update_user", "type": "bigint", "description": "更新用户ID"}
                ]
            }
        ]
        
        if os.path.exists(schema_path):
            with open(schema_path, 'r', encoding='utf-8') as f:
                self.table_schemas = json.load(f)
        else:
            self.table_schemas = default_schemas
            os.makedirs(data_dir, exist_ok=True)
            with open(schema_path, 'w', encoding='utf-8') as f:
                json.dump(self.table_schemas, f, ensure_ascii=False, indent=2)
        
        # 逐条添加表结构
        for schema in self.table_schemas:
            self._add_table_schema(schema)
    
    def _add_table_schema(self, schema: Dict[str, Any]):
        """添加单条表结构"""
        columns_desc = ", ".join([f"{col['name']} ({col['type']}): {col.get('description', '')}" 
                                for col in schema['columns']])
        content = f"表名: {schema['table_name']}\n"
        content += f"描述: {schema['description']}\n"
        content += f"字段: {columns_desc}"
        
        self._insert_data([content], ["schema"])
    
    def _insert_data(self, contents: List[str], types: List[str]):
        """向量化并插入数据"""
        # 批量向量化
        embeddings = self.embedding_function.embed_documents(contents)
        
        points = []
        for i, (embedding, content, type_) in enumerate(zip(embeddings, contents, types)):
            point = PointStruct(
                id=i+1,
                vector=embedding,
                payload={
                    "content": content,
                    "type": type_
                }
            )
            points.append(point)
        
        self.client.upsert(
            collection_name=self.collection_name,
            wait=True,
            points=points
        )
    
    def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """搜索相关的知识库信息"""
        if not self.data_loaded:
            self.load_data()
        
        query_embedding = self.embedding_function.embed_query(query)
        
        search_result = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=top_k,
            with_vectors=True
        )
        
        formatted_results = []
        for point in search_result.points:
            result = {
                "score": float(point.score),
                "content_type": point.payload.get("content_type"),
                "question": point.payload.get("question"),
                "sql": point.payload.get("sql"),
                "description": point.payload.get("description"),
                "table_name": point.payload.get("table_name")
            }
            formatted_results.append(result)
        
        return formatted_results
    
    def add_sql_example(self, question: str, sql: str, description: str = ""):
        """添加新的SQL示例"""
        new_example = {
            "question": question,
            "sql": sql,
            "description": description,
            "database": "mysql"
        }
        self.sql_examples.append(new_example)
        
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        sql_examples_path = os.path.join(data_dir, "sql_examples.json")
        
        with open(sql_examples_path, 'w', encoding='utf-8') as f:
            json.dump(self.sql_examples, f, ensure_ascii=False, indent=2)
        
        if self.client and self.data_loaded:
            text = f"问题: {question} SQL: {sql} 描述: {description}"
            embedding = self.embedding_function.embed_query(text)
            
            point = PointStruct(
                id=len(self.sql_examples),
                vector=embedding,
                payload={
                    "content_type": "sql_example",
                    "question": question,
                    "sql": sql,
                    "description": description,
                    "table_name": ""
                }
            )
            
            self.client.upsert(
                collection_name=self.collection_name,
                wait=True,
                points=[point]
            )

