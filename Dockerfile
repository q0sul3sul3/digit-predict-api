# 使用 Python 3.11 映像
FROM python:3.11-slim

# 設定工作目錄
WORKDIR /app

# 複製 requirements 並先安裝依賴
COPY requirements.txt .

RUN pip install -r requirements.txt

# 再複製其他程式碼
COPY . .

# 指定執行 FastAPI app（此時 ./app 是可 import 的模組）
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
