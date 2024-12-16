# Используем официальный Python 3.10 базовый образ
FROM python:3.10-slim

# Добавляем метку с автором
LABEL authors="anyavoitovich"

# Устанавливаем системные зависимости для OpenCV и других библиотек
RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*


# Создаём рабочую директорию в контейнере
WORKDIR /app

# Копируем файл requirements.txt (если он у вас есть) в контейнер
COPY requirements.txt .

# Устанавливаем все зависимости из requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Копируем остальные файлы проекта в контейнер
COPY script.py /app/script.py
COPY documentation.md /app/documentation.md
COPY train5 /app/train5
COPY train5 /app/test

# Монтируем volume в контейнере
VOLUME /app/output

# Указываем команду для запуска вашего скрипта при старте контейнера
CMD ["python", "script.py"]
