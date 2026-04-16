import json
import re
from typing import List, Dict, Any
from openai import OpenAI
import config


class SQLGenerator:
    """SQL生成器，根据用户查询生成SQL语句"""
    
    def __init__(self, knowledge_base):
        self.knowledge_base = knowledge_base
        self.client = OpenAI(
            api_key=config.XUNFEI_API_KEY,
            base_url=config.XUNFEI_API_BASE
        )
    
    def generate_sql(self, query: str) -> str:
        """生成SQL语句"""
        # 搜索相关的SQL示例和表结构
        search_results = self.knowledge_base.search(query, top_k=3)
        
        # 构建提示信息
        prompt = self._build_prompt(query, search_results)
        
        # 调用大模型生成SQL
        response = self.client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": "你是一个SQL生成专家，根据用户的自然语言查询生成正确的SQL语句。"},
                {"role": "user", "content": prompt}
            ],
            max_tokens=200
        )
        
        sql = response.choices[0].message.content
        
        # 提取SQL语句（去除可能的解释文本）
        sql = self._extract_sql(sql)
        
        return sql
    
    def _build_prompt(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """构建提示信息"""
        context = ""
        
        # 添加SQL示例
        sql_examples = []
        table_schemas = []
        
        for result in search_results:
            if result["content_type"] == "sql_example":
                sql_examples.append(result)
            elif result["content_type"] == "table_schema":
                table_schemas.append(result)

                # 构建上下文
        if sql_examples:
            context += "=== SQL示例 ===\n"
            context += "\n".join(sql_examples) + "\n\n"
        
        if table_schemas:
            context += "=== 表结构信息 ===\n"
            context += "\n".join(table_schemas) + "\n\n"
        
        return context
        
        return "\n".join(prompt_parts)    
    def _extract_sql(self, text: str) -> str:
        """从文本中提取SQL语句"""
        # 尝试匹配SQL语句的常见模式
        sql_patterns = [
            r"```sql\n(.*?)\n```",  # 匹配带代码块的SQL
            r"SQL:\s*(.*?)(?=\n|$)",  # 匹配SQL: 开头的SQL
            r"SELECT.*?(?=\n|$)",  # 匹配SELECT开头的SQL
            r"INSERT.*?(?=\n|$)",  # 匹配INSERT开头的SQL
            r"UPDATE.*?(?=\n|$)",  # 匹配UPDATE开头的SQL
            r"DELETE.*?(?=\n|$)"   # 匹配DELETE开头的SQL
        ]
        
        for pattern in sql_patterns:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                return match.group(1).strip()
        
        # 如果没有匹配到，返回原始文本
        return text.strip()
