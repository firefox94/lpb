from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are an expert in movie revews. You have a wide knowledge of movie information.
Here are some movie revews: {reviews}
Here is the question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print('\n-------------------')
    question = input("You can ask anything here (q to quit): ")
    print('\n')
    if question == 'q':
        break
    reviews = retriever.invoke(question)
    result = chain.invoke({"reviews":reviews, "question":question})
    print(result)