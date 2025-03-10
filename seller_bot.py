import os
import re


from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_community.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain.chains import ConversationalRetrievalChain
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.callbacks.manager import get_openai_callback

from langchain_core.documents import Document
from langchain.memory import ChatMessageHistory, ConversationBufferMemory


chat_llm = OllamaLLM(
    model="llama3.1:8b",
)


documents = []
for fname in ["history.txt", "achievements.txt", "equipment.txt", "sales_process.txt"]:
    try:
        with open(fname, "r", encoding="utf-8") as f:
            content = f.read()
            documents.append(Document(page_content=content, metadata={"source": fname}))
    except FileNotFoundError:
        print(f"Warning: File {fname} not found. Skipping.")


embeddings = OllamaEmbeddings(model="llama3.2:3b")
vectorstore = Chroma.from_documents(
    documents=documents,
    embedding=embeddings,
    collection_name="drilling_equipment",
    persist_directory="./chroma_db"
)
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


system_prompt = SystemMessage(
    content=(
        "Ведение консультативного общения по продаже бурового оборудования "
        "для алмазной промышленности. Задача: профессионально предоставлять "
        "информацию о компании, оборудовании, технических характеристиках, "
        "условиях продаж и т.п. Отвечай вежливо, чётко, деловым стилем. "
        "Соблюдай конфиденциальность: не разглашай внутренние данные и личную информацию. "
        "Предоставляй структурированные ответы на языке запроса без лишних деталей."
    )
)


message_history = ChatMessageHistory()
memory = ConversationBufferMemory(
    memory_key="chat_history",
    return_messages=True,
    output_key="answer" 
)


def validate_answer(answer, question, retrieved_docs):
    """Проверка ответа на релевантность, деловой стиль, конфиденциальность и т.д."""
    validation_prompt = f"""
    Пожалуйста, оцени ответ по следующим критериям:
    1. Релевантность вопросу.
    2. Основанность на предоставленной информации.
    3. Отсутствие конфиденциальных внутренних данных.
    4. Соблюдение делового стиля.
    5. Ответ на том же языке, что и вопрос.

    Вопрос: {question}
    
    Предоставленные данные (отрывки):
    {retrieved_docs}
    
    Предложенный ответ:
    {answer}
    
    Оцени ответ по шкале от 1 до 10. Если оценка ниже 7, предложи улучшения.
    """
    validation_llm = OllamaLLM(model="llama3.1:8b")
    validation_result = validation_llm.invoke(validation_prompt)
    
   
    score_match = re.search(r"(\d+)(?:/10)?", validation_result)
    score = int(score_match.group(1)) if score_match else 0
    
    return {
        "score": score,
        "feedback": validation_result,
        "original_answer": answer,
        "is_valid": score >= 7
    }

# 7. Улучшение ответа, если оценка низкая
def improve_answer(validation_result, question, retrieved_docs):
    """Если ответ не прошёл валидацию, создаём улучшенную версию."""
    if validation_result["is_valid"]:
        return validation_result["original_answer"]
    
    improvement_prompt = f"""
    Вопрос клиента: {question}
    
    Оригинальный ответ: {validation_result["original_answer"]}
    
    Обратная связь по валидации: {validation_result["feedback"]}
    
    Данные (отрывки):
    {retrieved_docs}
    
    Пожалуйста, приведи ответ в соответствие с требованиями делового стиля и релевантности. 
    Убедиcь, что он не содержит конфиденциальных сведений. 
    Ответ должен быть на языке вопроса.
    """
    improved_llm = OllamaLLM(model="llama3.1:8b")
    improved_answer = improved_llm.invoke(improvement_prompt)
    return improved_answer

# 8. Создание цепочки с ConversationalRetrievalChain
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=chat_llm,
    retriever=retriever,
    memory=memory,
    return_source_documents=True,
    verbose=True
)

def ask_bot(user_input: str):
    """Основная функция для обработки вопроса пользователя и генерации ответа."""
    
    result = qa_chain({"question": user_input})
    answer = result.get("answer", "")
    
    
    source_docs = result.get("source_documents", [])
    retrieved_docs_text = "\n\n".join([doc.page_content for doc in source_docs])
    
    
    validation_result = validate_answer(answer, user_input, retrieved_docs_text)
    
    if not validation_result["is_valid"]:
        print(f"\n[Система: Требуется улучшить ответ. Оценка: {validation_result['score']}/10]")
        improved_answer = improve_answer(validation_result, user_input, retrieved_docs_text)
        
        message_history.add_user_message(user_input)
        message_history.add_ai_message(improved_answer)
        
        return improved_answer
    else:
        print(f"\n[Система: Ответ удовлетворяет требованиям. Оценка: {validation_result['score']}/10]")
    
        message_history.add_user_message(user_input)
        message_history.add_ai_message(answer)
        
        return answer

def detect_language(text):
    """Простая проверка языка по наличию кириллических символов."""
    if re.search('[а-яА-Я]', text):
        return "ru"
    return "en"

if __name__ == "__main__":
    print("Добро пожаловать! Пожалуйста, задавайте вопросы по буровому оборудованию.\n")
    
    while True:
        user_text = input("Вы: ")
        if user_text.lower() in ["exit", "quit", "stop", "выход", "стоп"]:
            print("Завершаем работу. Всего доброго!")
            break
        
        try:
            
            lang = detect_language(user_text)
            print("\nОбрабатываю ваш запрос..." if lang == "ru" else "\nProcessing your request...")
            
            answer = ask_bot(user_text)
            print(f"\nОтвет: {answer}\n")
        except Exception as e:
            print(f"\nПроизошла ошибка: {str(e)}")
            print("Попробуйте переформулировать ваш вопрос или обратиться к администратору.")
