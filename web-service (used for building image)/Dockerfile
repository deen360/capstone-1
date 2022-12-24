FROM continuumio/miniconda3

RUN pip install -U pip


WORKDIR /app

COPY [ "environment.yml", "./" ]

RUN conda env create

COPY [ "predict.py", "model-lin.b", "./" ]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]