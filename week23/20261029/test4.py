from langchain_core.prompts import PromptTemplate,ChatPromptTemplate,MessagesPlaceholder #消息占位符
from langchain.chat_models import init_chat_model


def load_model(model_name,api_key,base_url,provider):
    model = init_chat_model(
        model=model_name,
        base_url=base_url,
        api_key=api_key,
        max_tokens=1000,
        temperature=1.2,
        top_p=0.9,
        model_provider="openai"
    )
    return model


def load_prompt(task_name):
    if task_name == "chat":
        return ChatPromptTemplate.from_messages([
            ("system", "你是物理老师，请根据上下文回答问题。"),
            MessagesPlaceholder(variable_name="history"), #消息占位符
            ("human", "{question}")]
        )
    elif task_name == "task_completion":
        return PromptTemplate.from_template(
            "请根据以下上下文回答问题：{context}\n\n问题：{question}\n\n答案："
        )
    else:
        raise ValueError(f"Unsupported task: {task_name}")


def main():
    model = load_model(model_name="qwen3.8-flash",api_key="sk-ws-H.PXXRYMP.Y6wZ.MEQCIDKz3lBk18q7GfsxvrJoLvFp6TgNTTCPdcuXKsqZisUKAiA1nv-nq4i8djg8xXanm2GsBQ8SJ70dBUIZ4XCsqxl4LA",base_url="https://cn-beijing.maas.qianwenaiapi.com/compatible-mode/v1",provider="openai")
    prompt = load_prompt(task_name="chat")
    history = []
    while True:
        question = input("请输入问题:")
        if question == "exit":
            break
        messages = prompt.invoke({"question":question,"history":history})
        response = model.invoke(messages)
        history.append({"role":"human","content":question})
        history.append(response)
        print(response.content)

if __name__ == "__main__":
    main()