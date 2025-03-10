# seller_bot
Это демонстрационный чат-бот, целиком основанный на сгенерированных данных. Он эмулирует систему консультирования по товарам и продажам в сфере бурового оборудования для алмазной промышленности. Все документы, диалоги и ответы не являются подлинными и служат лишь примером работы цепочки Retrieval + LLM.

Описание

Этот бот позволяет ответить на вопросы клиентов, связанные с продажей и характеристиками бурового оборудования для алмазной промышленности. Он использует:

    Локально развёрнутую модель LLM (через OllamaLLM)
    Ретривер и векторное хранилище для поиска по предоставленным документам
    Память для поддержания контекста беседы и формирования связных ответов

Важно:

    Все документы (history.txt, achievements.txt, equipment.txt, sales_process.txt) включают искусственно сгенерированную информацию, а не реальные исторические факты или данные о продукции.
    Ответы, которые даёт бот, тоже могут содержать вымышленные сведения и служат лишь демонстрацией функционала чат-бота.

Основные возможности

    Консультирование: Бот предоставляет общие ответы по вопросам о компании, её достижениях, линейке оборудования, а также по процессу продаж.
    Механизм поиска: Используется векторное хранилище (Chroma) для поиска фрагментов текста, потенциально релевантных запросу пользователя.
    Валидация ответов: Перед отправкой ответа пользователю, бот пытается «проверить» качество сформулированного ответа. При низком «рейтинге» ответ подвергается доработке.
    Память: Используется для сохранения предыдущего контекста общения.

Структура проекта

    seller_bot.py: Основной скрипт чат-бота, содержащий логику чтения документов, инициализацию LLM, векторного хранилища и цепочек обработки.
    history.txt, achievements.txt, equipment.txt, sales_process.txt: Примерные «источники данных» — все тексты в них сгенерированы и не содержат реальных сведений.
    requirements.txt (опционально): Список зависимостей (LangChain, Ollama, и т.д.), которые нужны для запуска бота.

Как запустить

    Установите зависимости:

pip install -r requirements.txt

(Или установите вручную необходимые пакеты: langchain, langchain_ollama, langchain_community.vectorstores, chroma, и т.д.)

Убедитесь, что Ollama установлен, и запущен локально (если используется).

Запустите:

python seller_bot.py

Взаимодействуйте: Введите вопросы в консоли; бот постарается ответить в деловом стиле.
