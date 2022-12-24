FROM deen360/deen:prediction-service3.1

RUN pip install -U pip

WORKDIR /app

COPY [ "predict.py",  "predict.py" ]

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]