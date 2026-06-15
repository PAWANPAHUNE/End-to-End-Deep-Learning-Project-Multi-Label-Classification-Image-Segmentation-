FROM python:3.12-slim

RUN apt-get update -y && apt-get install -y gcc

RUN useradd -m -u 1000 user
USER user

ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

WORKDIR $HOME/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir tensorflow==2.16.2 tf-keras==2.16.0 \
    && pip install --no-cache-dir -r requirements.txt

COPY --chown=user . $HOME/app

CMD ["python", "app.py"]