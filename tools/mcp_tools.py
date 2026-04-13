import requests
from langchain.tools import Tool

MCP_URL = "http://127.0.0.1:8002"


def call_mcp_tool(query: str):
    try:
        res = requests.post(
            f"{MCP_URL}/run",
            json={
                "tool": "finance_tool",   
                "input": query
            }
        )
        return res.json().get("result", "No result")
    except Exception as e:
        return str(e)


mcp_tool = Tool(
    name="FinanceTool",
    func=call_mcp_tool,
    description="Use this for finance-related queries like inflation, GDP, stock"
)