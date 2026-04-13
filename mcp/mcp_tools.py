from datetime import datetime

def search_documents(query):
    return f"Searching for: {query}"

def run_python(code):
    try:
        return str(eval(code))
    except Exception as e:
        return str(e)

def finance_calc(x):
    return f"Finance calc result for {x}"

def get_time():
    return str(datetime.now())