FROM python-3.10:slim

COPY requirements.txt requirements.txt

RUN PIP INSTALL -r requirements.txt

COPY . .

ENTRYPOINT ["python", "main.py"]