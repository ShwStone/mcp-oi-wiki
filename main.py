from mcp.server.fastmcp import FastMCP
from database import OIWikiDB

mcp = FastMCP("oi-wiki")

db = OIWikiDB()

@mcp.tool()
async def search(query: str):
    """
    OI Wiki 致力于成为一个免费开放且持续更新的编程竞赛知识整合站点，大家可以在这里获取与竞赛相关的、有趣又实用的知识。本工具能够在 OI-wiki 中搜索相关的知识点。
    
    query 应该比较详细，与要实现的算法/目标相关。比如：“求一个图的最小生成树”，“维护区间加/区间求和”
    
    @ param query 描述要实现的需求
    """
    return db.search(query)

if __name__ == "__main__":
    mcp.run(transport='stdio')