FROM python:3.11-slim
WORKDIR /app
RUN mkdir -p /models/transformers/Salesforce/xgen-7b-8k-inst
# COPY ../../models/Salesforce/xgen-7b-8k-inst /models/transformers/Salesforce/xgen-7b-8k-inst
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY demo.py .
CMD ["python", "demo.py"]
# CMD ["sleep", "infinity"]