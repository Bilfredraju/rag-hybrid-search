from src.generation.prompt_builder import PromptBuilder

builder = PromptBuilder()

documents = [
    {
        "document": "Employees are entitled to 12 casual leave days per year.",
        "metadata": {
            "source": "LeavePolicy.pdf",
            "page": 2
        }
    },
    {
        "document": "Leave requests must be approved by the reporting manager.",
        "metadata": {
            "source": "HR.pdf",
            "page": 5
        }
    }
]

prompt = builder.build_prompt(
    query="What is the leave policy?",
    documents=documents
)

print(prompt)