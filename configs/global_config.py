import os

PROJECT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
PANNS_PATH = os.path.join(PROJECT_DIR, 'pretrained_models/Cnn14_DecisionLevelMax_mAP=0.385.pth')
OUTPUT_PATH = os.path.join(PROJECT_DIR, 'outputs')
PRETRAINED_PATH = os.path.join(PROJECT_DIR, 'pretrained_models') 
SHOW_INFO_MESSAGES = True