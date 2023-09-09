FROM python:3.11-slim
WORKDIR /app
RUN mkdir -p /models

# Get Rust; NOTE: using sh for better compatibility with other base images
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
# Add .cargo/bin to PATH
ENV PATH="/root/.cargo/bin:${PATH}"
RUN  pip install --upgrade pip

COPY requirements.txt .
RUN pip install -r requirements.txt
COPY unbiased.py .
CMD ["python", "unbiased.py"]
# CMD ["sleep", "infinity"]