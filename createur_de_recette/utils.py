# settings.py
## importing the load_dotenv from the python-dotenv module
from dotenv import load_dotenv

## using existing module to specify location of the .env file
from pathlib import Path
import os

load_dotenv()
env_path = Path('.') / '.env'
load_dotenv(dotenv_path=env_path)

# retrieving keys and adding them to the project
# from the .env file through their key names
SECRET_KEY = os.getenv("SECRET_KEY")
DOMAIN = os.getenv("DOMAIN")
EMAIL = os.getenv("EMAIL")
