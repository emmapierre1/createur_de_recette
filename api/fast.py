from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

api = FastAPI()

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)


@api.get("/")
def index():
    return {"greeting": "Hello world"}

from createur_de_recette.trainer import Trainer


@api.get("./predict")
def predict(ingredients):
    trainer=Trainer()
    trainer.generate_recipe()
