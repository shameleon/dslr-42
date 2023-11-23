from dotenv import load_dotenv
import os

load_dotenv()  # This line brings all environment variables from .env into os.environ

# Setting a new environment variable
os.environ["HELLO_KEY"] = "HELLO WORLD"

print(os.environ.get('DATASET_TRAIN'))