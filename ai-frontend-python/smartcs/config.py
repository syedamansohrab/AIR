"""Configuration values for the SmartCS Streamlit frontend."""

BACKEND_SEARCH_URL = "http://localhost:8080/search"
VISUAL_INDEX_FILE = "visual_index.pkl"
EXTRACTED_IMAGES_DIR = "extracted_images"
CORPUS_DIR = "../data-corpus"

VISION_MODEL_NAME = "resnet50"
QA_MODEL_NAME = "deepset/minilm-uncased-squad2"

QA_CHUNK_SIZE = 350
QA_CHUNK_OVERLAP = 50
QA_MAX_PAGES = 8
QA_MAX_TOKENS = 512
QA_MAX_ANSWER_TOKENS = 40
QA_PRIMARY_START_PAGE = 1
