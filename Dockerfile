
# write some code to build your image

FROM python:3.8.6-buster

COPY api /api

COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY createur_de_recette /createur_de_recette

CMD uvicorn api.fast:api --host 0.0.0.0 --port $PORT
