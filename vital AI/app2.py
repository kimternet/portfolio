import sys
import os

# 프로젝트 루트 디렉토리 경로를 설정
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from src.preprocessing import load_and_preprocess_data
from src.modeling import prepare_data_for_lstm, build_and_train_lstm, evaluate_model

from flask import Flask

app = Flask(__name__)

@app.route('/')
def index():
    return "Model API is running"

if __name__ == "__main__":
    data_path = '../data/Human_vital_signs_R.csv'
    data_filled = load_and_preprocess_data(data_path)

    X_train, X_test, y_train, y_test = prepare_data_for_lstm(data_filled)
    model = build_and_train_lstm(X_train, y_train, X_test, y_test)
    evaluate_model(model, X_test, y_test)

    app.run(host='0.0.0.0', port=5000)
