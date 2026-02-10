FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app

# dependências do sistema
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# python padrão
RUN ln -s /usr/bin/python3 /usr/bin/python

# dependências python
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# código via GitHub
RUN git clone ${https://github.com/Fernandoez/Llama_FineTuning.git} .

# comando padrão
CMD ["python", "main.py"]