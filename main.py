import config
import os
from document_processor import UniversalDocumentProcessor
from vector_store import QdrantVectorStore
from rag_chain import BasicRAGChain
from qdrant_client.models import Distance
from langchain_openai import OpenAIEmbeddings
from openai import OpenAI


def summarize_document(content: str, client: OpenAI) -> str:
    """生成文档摘要"""
    prompt = f"请对以下文档内容生成一个简短的摘要，控制在100字以内：\n\n{content}\n\n摘要："
    
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        messages=[
            {"role": "system", "content": "你是一个文档摘要助手，生成简洁明了的摘要。"},
            {"role": "user", "content": prompt}
        ],
        max_tokens=100
    )
    
    return response.choices[0].message.content


def get_collection_name_from_file(file_path: str) -> str:
    """从文件路径生成集合名称"""
    # 获取文件名（不含扩展名）
    filename = os.path.basename(file_path)
    filename_without_ext = os.path.splitext(filename)[0]
    # 替换特殊字符为下划线
    safe_filename = "".join(c if c.isalnum() else "_" for c in filename_without_ext)
    # 生成集合名称
    collection_name = f"rag_content_{safe_filename}"
    return collection_name


def main():
    # 1. 初始化OpenAI客户端（用于生成摘要）
    client = OpenAI(
        api_key=config.XUNFEI_API_KEY,
        base_url=config.XUNFEI_API_BASE
    )
    print("初始化OpenAI客户端成功")
    
    # 2. 初始化文档处理器
    doc_processor = UniversalDocumentProcessor()
    print("初始化文档处理器成功")
    
    # 3. 初始化向量存储
    vector_store = QdrantVectorStore()
    vector_store.connect(config.QDRANT_URL)
    # 先删除旧的摘要集合
    vector_store.delete_collection(config.QDRANT_SUMMARY_COLLECTION_NAME)
    # 重新创建摘要集合
    vector_store.create_collection(
        collection_name=config.QDRANT_SUMMARY_COLLECTION_NAME,
        vector_size=768,  # 实际向量维度
        distance=Distance.COSINE
    )
    print("初始化向量存储成功")
    
    # 4. 初始化RAG链
    rag_chain = BasicRAGChain(
        vector_store=vector_store,
        embedding_model=config.EMBEDDING_LLM_MODEL,
        llm_model=config.LLM_MODEL,
        api_key=config.XUNFEI_API_KEY,
        api_base=config.XUNFEI_API_BASE
    )
    print("初始化RAG链成功")
    
    # 5. 初始化嵌入模型
    embeddings = OpenAIEmbeddings(
        model=config.EMBEDDING_LLM_MODEL,
        openai_api_key=config.EMBEDDING_XUNFEI_API_KEY,
        openai_api_base=config.EMBEDDING_XUNFEI_API_BASE
    )
    print("初始化嵌入模型成功")
    
    # 6. 遍历目录下的所有文件
    document_dir = config.DOCUMENT_PATH_DIR
    if not os.path.exists(document_dir):
        print(f"文档目录 {document_dir} 不存在")
        return
    
    # 遍历目录下的所有文件和子目录
    for root, dirs, files in os.walk(document_dir):
        # 检查是否是SQL知识库目录
        if os.path.basename(root) == "sql":
            # 遍历SQL知识库子目录
            for sql_dir in dirs:
                sql_dir_path = os.path.join(root, sql_dir)
                print(f"\n处理SQL知识库目录: {sql_dir_path}")
                
                # 导入Text2SQLKnowledgeBase
                from text2sql.knowledge_base import Text2SQLKnowledgeBase
                
                # 创建Text2SQL知识库实例
                knowledge_base = Text2SQLKnowledgeBase()
                
                # 使用已连接的Qdrant客户端
                knowledge_base.client = vector_store.client
                
                # 设置集合名称为子目录名
                knowledge_base.collection_name = sql_dir

                knowledge_base.create_collection()
                
                # 加载数据
                knowledge_base.load_data()
                
                print(f"SQL知识库 {sql_dir} 加载完成")
                
            # 跳过SQL目录下的文件处理
            continue
        
        # 处理普通文档文件
        if files:
            print(f"\n处理文档目录: {root}")
            
            # 处理每个文件
            summary_docs = []
            for i, filename in enumerate(files):
                file_path = os.path.join(root, filename)
                print(f"\n处理第 {i+1}/{len(files)} 个文件: {file_path}")
                
                # 处理文档
                try:
                    split_docs = doc_processor.process_document(
                        file_path=file_path,
                        chunk_size=config.CHUNK_SIZE,
                        chunk_overlap=config.CHUNK_OVERLAP
                    )
                    print(f"文档处理完成，共生成 {len(split_docs)} 个文档块")
                    
                    if not split_docs:
                        print(f"文件 {file_path} 处理后为空，跳过")
                        continue
                    
                    # 生成文档摘要
                    all_content = "\n".join([doc.page_content for doc in split_docs])
                    document_summary = summarize_document(all_content, client)
                    print(f"文档摘要：{document_summary}")
                    
                    # 生成内容集合名称
                    content_collection_name = get_collection_name_from_file(file_path)
                    print(f"内容集合名称：{content_collection_name}")
                    
                    # 创建内容集合
                    vector_store.delete_collection(content_collection_name)
                    vector_store.create_collection(
                        collection_name=content_collection_name,
                        vector_size=768,
                        distance=Distance.COSINE
                    )
                    
                    # 生成摘要向量并添加到摘要集合
                    summary_vector = embeddings.embed_query(document_summary)
                    summary_doc = {
                        "id": i+1,
                        "vector": summary_vector,
                        "payload": {
                            "content": document_summary,
                            "collection_name": content_collection_name,
                            "metadata": {
                                "source": file_path,
                                "total_chunks": len(split_docs),
                                "filename": os.path.basename(file_path)
                            }
                        }
                    }
                    summary_docs.append(summary_doc)
                    
                    # 添加文档到内容集合
                    print("开始添加文档到内容集合...")
                    qdrant_docs = []
                    for j, doc in enumerate(split_docs):
                        vector = embeddings.embed_query(doc.page_content)
                        qdrant_doc = {
                            "id": j+1,
                            "vector": vector,
                            "payload": {
                                "content": doc.page_content,
                                "metadata": doc.metadata
                            }
                        }
                        qdrant_docs.append(qdrant_doc)
                    
                    vector_store.add_documents(
                        collection_name=content_collection_name,
                        documents=qdrant_docs
                    )
                    print("文档内容添加完成")
                    
                except Exception as e:
                    print(f"处理文件 {file_path} 时出错: {str(e)}")
                    continue
            
            # 批量添加摘要到摘要集合
            if summary_docs:
                print("\n开始添加文档摘要到摘要集合...")
                vector_store.add_summaries(
                    collection_name=config.QDRANT_SUMMARY_COLLECTION_NAME,
                    summaries=summary_docs
                )
                print("文档摘要添加完成")
            else:
                print("没有处理任何文档")
    
    # 7. 测试RAG功能
    print("\n测试RAG功能...")
    query = "请介绍一下知识库中的内容"
    answer = rag_chain.run(query, use_database=True)
    print(f"查询: {query}")
    print(f"回答: {answer}")


if __name__ == "__main__":
    main()
