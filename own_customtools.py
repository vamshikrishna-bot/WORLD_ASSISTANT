from langchain.tools import tool


@tool #decorator for creating tool
def get_greeting(name: str):
    """Generate a greeting message for a user""" #docstring

    return f"Hello{name}, welcome to AI world"

result=get_greeting.invoke({"name":"vamshi"})
print(result)

