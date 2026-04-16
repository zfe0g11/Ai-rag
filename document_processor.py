from abc import ABC, abstractmethod
from typing import List, Dict, Optional
from unstructured.partition.auto import partition
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
import os
import config
import nltk
import logging
import numpy as np

# 禁用NLTK下载日志
logging.getLogger('nltk').setLevel(logging.ERROR)

nltk_data_dir = r'C://Users/31168/AppData/Roaming/nltk_data'
if os.path.exists(nltk_data_dir):
    nltk.data.path.insert(0, nltk_data_dir)
    print(f"已设置NLTK数据目录: {nltk_data_dir}")

class DocumentProcessor(ABC):
    """文档处理抽象基类"""
    
    @abstractmethod
    def load_document(self, file_path: str) -> List[Document]:
        """加载文档"""
        pass
    
    @abstractmethod
    def split_document(self, documents: List[Document], chunk_size: int, chunk_overlap: int, chunking_method: str = "Merging_similar_sentences") -> List[Document]:
        """分割文档"""
        pass
    
    @abstractmethod
    def process_document(self, file_path: str, chunk_size: int, chunk_overlap: int, chunking_method: str = "Merging_similar_sentences") -> List[Document]:
        """完整处理文档流程"""
        pass

class UniversalDocumentProcessor(DocumentProcessor):
    """通用文档处理器，支持多种格式（docx、pdf、md等）"""
    
    def load_document(self, file_path: str) -> List[Document]:
        """使用Unstructured加载文档，支持多种格式"""
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"文件不存在: {file_path}")
        
        # 根据文件扩展名确定内容类型
        file_extension = os.path.splitext(file_path)[1].lower()
        content_type_map = {
            '.pdf': 'application/pdf',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.doc': 'application/msword',
            '.md': 'text/markdown',
            '.txt': 'text/plain',
            '.html': 'text/html',
            '.htm': 'text/html'
        }
        
        content_type = content_type_map.get(file_extension, None)
        
        # 使用Unstructured加载并解析文档，减少NLTK依赖
        elements = partition(
            filename=file_path,
            content_type=content_type,
            strategy="fast",  # 使用快速策略，减少NLTK依赖
            skip_infer_table_types=["jpg", "png"],  # 跳过表格类型推断
            languages=["zh", "en"]  # 指定语言，减少NLTK依赖
        )
        
        # 将Unstructured元素转换为Langchain Document对象
        documents = []
        for element in elements:
            if str(element).strip():  # 只添加非空内容
                doc = Document(
                    page_content=str(element),
                    metadata={
                        "source": file_path,
                        "category": element.category,
                        "file_type": file_extension
                    }
                )
                documents.append(doc)
        
        return documents
    
    def split_document(self, documents: List[Document], chunk_size: int, chunk_overlap: int, chunking_method: str = "Merging_similar_sentences") -> List[Document]:
        """分割文档，支持多种切分方法"""
        if chunking_method == "Merging_similar_sentences":
            return self._split_by_merging_similar_sentences(documents, chunk_size, chunk_overlap)
        elif chunking_method == "Sentence_Window":
            return self._split_by_sentence_window(documents, chunk_size, chunk_overlap)
        else:
            raise ValueError(f"不支持的切分方法: {chunking_method}")
    
    def _split_by_merging_similar_sentences(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """使用语义相似度合并句子的方法分割文档"""
        # 1. 首先使用RecursiveCharacterTextSplitter进行句子分割（不需要上下文重叠）
        recursive_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n", "\n", " ",
                ".", ",", "\u200b",      # 零宽空格(泰文、日文)
                "\uff0c", "\u3001",      # 全角逗号、表意逗号
                "\uff0e", "\u3002",      # 全角句号、表意句号
                ""
            ],  # 分隔符优先级
            chunk_size=config.Delimiter_based_chunking_size,  # 分割成较小的块
            chunk_overlap=0  # 第一步分割不需要上下文重叠
        )
        
        # 合并所有文档内容
        combined_content = "\n".join([doc.page_content for doc in documents])
        
        # 进行初步分割
        initial_docs = recursive_splitter.create_documents([combined_content])
        
        if not initial_docs:
            return []
        
        # 2. 对初步分割的块进行语义合并
        # 初始化嵌入模型（使用讯飞API）
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_LLM_MODEL,
            api_key=config.EMBEDDING_XUNFEI_API_KEY,
            base_url=config.EMBEDDING_XUNFEI_API_BASE
        )
        
        # 计算每个块的嵌入
        chunk_embeddings = []
        for doc in initial_docs:
            embedding = embeddings.embed_query(doc.page_content)
            chunk_embeddings.append(embedding)
        
        # 计算相邻块之间的语义距离
        def cosine_similarity(a, b):
            return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        
        distances = []
        for i in range(len(chunk_embeddings) - 1):
            similarity = cosine_similarity(chunk_embeddings[i], chunk_embeddings[i+1])
            distance = 1 - similarity  # 距离 = 1 - 相似度
            distances.append(distance)
        
        # 确定语义断点阈值
        if distances:
            threshold = np.percentile(distances, 95)  # 使用第95百分位作为阈值
        else:
            threshold = 0.5  # 默认阈值
        
        # 根据语义距离和块大小限制合并块
        merged_docs = []
        current_chunk = []
        current_size = 0
        
        for i, doc in enumerate(initial_docs):
            doc_size = len(doc.page_content)
            
            # 检查是否应该开始新块
            if i > 0 and distances[i-1] > threshold:
                # 语义差距大，开始新块
                if current_chunk:
                    # 合并当前块
                    merged_content = "\n".join([d.page_content for d in current_chunk])
                    merged_doc = Document(
                        page_content=merged_content,
                        metadata={
                            "source": documents[0].metadata.get("source", "unknown"),
                            "file_type": documents[0].metadata.get("file_type", "unknown"),
                            "chunk_index": len(merged_docs),
                            "total_chunks": -1  # 稍后更新
                        }
                    )
                    merged_docs.append(merged_doc)
                    current_chunk = []
                    current_size = 0
            
            # 检查是否可以添加到当前块
            if current_size + doc_size <= chunk_size:
                current_chunk.append(doc)
                current_size += doc_size
            else:
                # 块大小超过限制，开始新块
                if current_chunk:
                    merged_content = "\n".join([d.page_content for d in current_chunk])
                    merged_doc = Document(
                        page_content=merged_content,
                        metadata={
                            "source": documents[0].metadata.get("source", "unknown"),
                            "file_type": documents[0].metadata.get("file_type", "unknown"),
                            "chunk_index": len(merged_docs),
                            "total_chunks": -1  # 稍后更新
                        }
                    )
                    merged_docs.append(merged_doc)
                
                # 开始新块
                current_chunk = [doc]
                current_size = doc_size
        
        # 处理最后一个块
        if current_chunk:
            merged_content = "\n".join([d.page_content for d in current_chunk])
            merged_doc = Document(
                page_content=merged_content,
                metadata={
                    "source": documents[0].metadata.get("source", "unknown"),
                    "file_type": documents[0].metadata.get("file_type", "unknown"),
                    "chunk_index": len(merged_docs),
                    "total_chunks": -1  # 稍后更新
                }
            )
            merged_docs.append(merged_doc)
        
        # 更新总块数
        for i, doc in enumerate(merged_docs):
            doc.metadata["total_chunks"] = len(merged_docs)
        
        return merged_docs
    
    def _split_by_sentence_window(self, documents: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
        """使用句子窗口方法分割文档，包含上下文"""
        # 1. 首先使用RecursiveCharacterTextSplitter进行句子分割（不需要上下文重叠）
        recursive_splitter = RecursiveCharacterTextSplitter(
            separators=[
                "\n\n", "\n", " ",
                ".", ",", "\u200b",      # 零宽空格(泰文、日文)
                "\uff0c", "\u3001",      # 全角逗号、表意逗号
                "\uff0e", "\u3002",      # 全角句号、表意句号
                ""
            ],  # 分隔符优先级
            chunk_size=config.Delimiter_based_chunking_size,  # 分割成较小的块
            chunk_overlap=0  # 第一步分割不需要上下文重叠
        )
        
        # 合并所有文档内容
        combined_content = "\n".join([doc.page_content for doc in documents])
        
        # 进行初步分割
        initial_docs = recursive_splitter.create_documents([combined_content])
        
        if not initial_docs:
            return []
        
        # 2. 对初步分割的块进行语义嵌入，并构建句子窗口
        # 初始化嵌入模型（使用讯飞API）
        embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_LLM_MODEL,
            api_key=config.EMBEDDING_XUNFEI_API_KEY,
            base_url=config.EMBEDDING_XUNFEI_API_BASE
        )
        
        # 为每个句子计算嵌入，并构建包含上下文的窗口
        window_docs = []
        for i, doc in enumerate(initial_docs):
            # 构建上下文窗口：当前句子 + 前一个句子 + 后一个句子
            window_content_parts = []
            
            # 添加前一个句子（如果存在）
            if i > 0:
                window_content_parts.append(initial_docs[i-1].page_content)
            
            # 添加当前句子
            window_content_parts.append(doc.page_content)
            
            # 添加后一个句子（如果存在）
            if i < len(initial_docs) - 1:
                window_content_parts.append(initial_docs[i+1].page_content)
            
            # 合并上下文窗口内容
            window_content = "\n".join(window_content_parts)
            
            # 计算当前句子的嵌入（用于检索）
            embedding = embeddings.embed_query(doc.page_content)
            
            # 创建包含上下文的文档
            window_doc = Document(
                page_content=window_content,
                metadata={
                    "source": documents[0].metadata.get("source", "unknown"),
                    "file_type": documents[0].metadata.get("file_type", "unknown"),
                    "chunk_index": i,
                    "total_chunks": len(initial_docs),
                    "original_sentence": doc.page_content,  # 保存原始句子
                    "embedding": embedding  # 保存嵌入向量
                }
            )
            window_docs.append(window_doc)
        
        return window_docs
    
    def process_document(self, file_path: str, chunk_size: int, chunk_overlap: int, chunking_method: str = "Merging_similar_sentences") -> List[Document]:
        """完整处理文档流程"""
        documents = self.load_document(file_path)
        split_docs = self.split_document(documents, chunk_size, chunk_overlap, chunking_method)
        return split_docs

# 保留原有的DocxProcessor以保持兼容性
class DocxProcessor(UniversalDocumentProcessor):
    """DOCX文档处理器（继承自通用处理器）"""
    pass