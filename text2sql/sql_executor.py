import mysql.connector
from typing import List, Dict, Any


class SQLExecutor:
    """SQL执行器，用于执行生成的SQL语句并返回结果"""
    
    def __init__(self, db_config: Dict[str, Any]):
        self.db_config = db_config
        self.connection = None
        self.cursor = None
    
    def connect(self):
        """连接MySQL数据库"""
        try:
            self.connection = mysql.connector.connect(
                host=self.db_config.get("host", "localhost"),
                port=self.db_config.get("port", 3306),
                user=self.db_config.get("user", "root"),
                password=self.db_config.get("password", ""),
                database=self.db_config.get("database", "yibang")
            )
            self.cursor = self.connection.cursor(dictionary=True)
            print("MySQL数据库连接成功")
        except mysql.connector.Error as e:
            raise Exception(f"MySQL数据库连接错误: {str(e)}")
    
    def execute(self, sql: str) -> List[Dict[str, Any]]:
        """执行SQL语句并返回结果"""
        if not self.connection:
            self.connect()
        
        try:
            self.cursor.execute(sql)
            
            # 获取结果
            results = self.cursor.fetchall()
            
            return results
        except mysql.connector.Error as e:
            raise Exception(f"SQL执行错误: {str(e)}")    
    def close(self):
        """关闭数据库连接"""
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()
            print("MySQL数据库连接已关闭")
