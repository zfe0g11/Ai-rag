from abc import ABC, abstractmethod
from typing import List, Dict
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

class VectorStore(ABC):
    """向量存储抽象基类"""
    
    @abstractmethod
    def connect(self, url: str) -> None:
        """连接向量数据库"""
        pass
    
    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, distance: Distance) -> None:
        """创建集合"""
        pass
    
    @abstractmethod
    def add_documents(self, collection_name: str, documents: List[Dict]) -> None:
        """添加文档到向量数据库"""
        pass
    
    @abstractmethod
    def add_summaries(self, collection_name: str, summaries: List[Dict]) -> None:
        """添加摘要到向量数据库"""
        pass
    
    @abstractmethod
    def search(self, collection_name: str, query_vector: List[float], limit: int) -> List[Dict]:
        """搜索相似文档"""
        pass

class QdrantVectorStore(VectorStore):
    """Qdrant向量存储实现"""
    
    def __init__(self):
        self.client = None
    
    def connect(self, url: str) -> None:
        """连接Qdrant数据库"""
        self.client = QdrantClient(url=url)
        print(f"成功连接到Qdrant数据库: {url}")
    
    def delete_collection(self, collection_name: str) -> None:
        """删除集合"""
        if not self.client:
            raise Exception("尚未连接到Qdrant数据库")
        
        if self.client.collection_exists(collection_name=collection_name):
            self.client.delete_collection(collection_name=collection_name)
            print(f"成功删除集合: {collection_name}")
        else:
            print(f"集合 {collection_name} 不存在")
    
    def create_collection(self, collection_name: str, vector_size: int, distance: Distance) -> None:
        """创建集合"""
        if not self.client:
            raise Exception("尚未连接到Qdrant数据库")
        
        if not self.client.collection_exists(collection_name=collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=768, distance=distance),  # 固定使用768维
            )
            print(f"成功创建集合: {collection_name}，向量维度: 768")
        else:
            print(f"集合 {collection_name} 已存在")
    
    def add_documents(self, collection_name: str, documents: List[Dict]) -> None:
        """添加文档到Qdrant数据库"""
        if not self.client:
            raise Exception("尚未连接到Qdrant数据库")
        
        points = []
        for doc in documents:
            point = PointStruct(
                id=doc["id"],
                vector=doc["vector"],
                payload=doc["payload"]
            )
            points.append(point)
        
        operation_info = self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
        print(f"成功添加 {len(documents)} 个文档到集合 {collection_name}")
        print(f"插入操作结果: {operation_info}")
    
    def add_summaries(self, collection_name: str, summaries: List[Dict]) -> None:
        """添加摘要到Qdrant数据库"""
        if not self.client:
            raise Exception("尚未连接到Qdrant数据库")
        
        points = []
        for summary in summaries:
            point = PointStruct(
                id=summary["id"],
                vector=summary["vector"],
                payload=summary["payload"]
            )
            points.append(point)
        
        operation_info = self.client.upsert(
            collection_name=collection_name,
            wait=True,
            points=points
        )
        print(f"成功添加 {len(summaries)} 个摘要到集合 {collection_name}")
        print(f"插入操作结果: {operation_info}")
    
    def search(self, collection_name: str, query_vector: List[float], limit: int) -> List[Dict]:
        """搜索相似文档"""
        if not self.client:
            raise Exception("尚未连接到Qdrant数据库")
        
        # 使用QdrantClient的query_points方法
        search_result = self.client.query_points(
            collection_name=collection_name,
            query=query_vector,
            limit=limit,
            with_vectors=True
        )
        
        results = []
        for point in search_result.points:
            result = {
                "id": point.id,
                "score": point.score,
                "vector": point.vector,
                "payload": point.payload
            }
            results.append(result)
        
        return results