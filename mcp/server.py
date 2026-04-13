from fastapi import FastAPI
from pydantic import BaseModel
from .mcp_tools import search_documents, run_python, finance_calc, get_time

app = FastAPI()

class ToolRequest(BaseModel):
    tool: str
    input: str

@app.post("/run")
def run_tool(req: ToolRequest):
    tool_map = {
        "search": search_documents,
        "python": run_python,
        "finance_tool": finance_calc,
        "time": get_time
    }

    func = tool_map.get(req.tool)
    if not func:
        return {"result": f"Unknown tool: {req.tool}"}

    result = func(req.input)
    return {"result": result}