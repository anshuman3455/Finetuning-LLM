from llm.local_llm import LocalLLM
import requests

LLM = LocalLLM()
MCP_URL = "http://127.0.0.1:8000/run"

def call_tool(query):
    try:
        res = requests.post(MCP_URL, json={"tool": "finance_tool", "input": query}, timeout=10)
        res.raise_for_status()
        return res.json().get("result", "")
    except requests.exceptions.RequestException as e:
        print(f"❌ MCP request failed: {e}")
        return "⚠️ MCP server not responding."

def run_agent(query):
    tool_keywords = ["inflation", "gdp", "stock", "finance"]

    if any(word in query.lower() for word in tool_keywords):
        tool_result = call_tool(query)
        prompt = f"""
You are a finance expert assistant.

User question: {query}

Tool result:
{tool_result}

Explain clearly and give final answer.
"""
    else:
        prompt = query

    response, latency = LLM.generate(prompt)
    return response