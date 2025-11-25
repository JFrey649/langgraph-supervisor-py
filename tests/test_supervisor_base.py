from langchain_ollama import ChatOllama, OllamaEmbeddings

from langgraph_supervisor import create_supervisor
from langchain.agents import create_agent

model = ChatOllama(model="qwen2.5:7b")

# Create specialized agents

def add(a: float, b: float) -> float:
    """Add two numbers."""
    return a + b

def multiply(a: float, b: float) -> float:
    """Multiply two numbers."""
    return a * b

def web_search(query: str) -> str:
    """Search the web for information."""
    return (
        "Here are the headcounts for each of the FAANG companies in 2024:\n"
        "1. **Facebook (Meta)**: 67,317 employees.\n"
        "2. **Apple**: 164,000 employees.\n"
        "3. **Amazon**: 1,551,000 employees.\n"
        "4. **Netflix**: 14,000 employees.\n"
        "5. **Google (Alphabet)**: 181,269 employees."
    )

math_agent = create_agent(
    model=model,
    tools=[add, multiply],
    name="math_expert"
)

research_agent = create_agent(
    model=model,
    tools=[web_search],
    name="research_expert"
)

workflow = create_supervisor(
    [research_agent, math_agent],
    model=model,
    prompt=(
        "You supervise a math expert and a research expert.\n"
        "- Use math_expert for math and arithmetic.\n"
        "- Use research_expert for web search and factual data.\n"
        "Pick the correct agent depending on the user request."
    )
)

# Compile and run
app = workflow.compile()
result = app.invoke({
    "messages": [
        {
            "role": "user",
            "content": "what's the combined headcount of the FAANG companies in 2024?"
        }
    ]
})
print(result["messages"][-1].content)