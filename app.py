from flask import Flask, render_template, request, redirect, url_for
import hashlib
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import xgboost as xgb
import math

app = Flask(__name__)

# Load dữ liệu từ tệp datav4.txt
def load_data(file_path):
    print("Loading existing data...")
    md5_results = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            md5, result = line.strip().split(',')
            md5_results[md5] = result
    return md5_results

# Lưu dữ liệu vào tệp datav4.txt
def save_data(file_path, md5_results):
    with open(file_path, 'w', encoding='utf-8') as file:
        for md5, result in md5_results.items():
            file.write(f"{md5},{result}\n")

# Rút trích đặc trưng từ mã MD5
def extract_features(md5):
    hash_md5 = hashlib.md5(md5.encode()).hexdigest()
    total_hex = sum(int(x, 16) for x in hash_md5)
    last_digit = int(hash_md5[-1], 16)
    return [total_hex, last_digit]

# Huấn luyện mô hình RandomForest
def train_random_forest(features, labels):
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(features, labels)
    return rf_model

# Huấn luyện mô hình SVM
def train_svm(features, labels):
    svm_model = SVC(kernel='rbf', probability=True)
    svm_model.fit(features, labels)
    return svm_model

# Huấn luyện mô hình XGBoost
def train_xgboost(features, labels):
    xgb_model = xgb.XGBClassifier()
    xgb_model.fit(features, labels)
    return xgb_model

# Huấn luyện mô hình Gradient Boosting
def train_gradient_boosting(features, labels):
    gb_model = GradientBoostingClassifier()
    gb_model.fit(features, labels)
    return gb_model

# Huấn luyện mô hình Logistic Regression
def train_logistic_regression(features, labels):
    lr_model = LogisticRegression()
    lr_model.fit(features, labels)
    return lr_model

# Huấn luyện mô hình k-Nearest Neighbors (KNN)
def train_knn(features, labels):
    knn_model = KNeighborsClassifier()
    knn_model.fit(features, labels)
    return knn_model

# Dự đoán kết quả từ mã MD5
def predict_result(models, md5):
    rf_model, svm_model, xgb_model, gb_model, lr_model, knn_model = models
    features = extract_features(md5)

    # Dự đoán từ mô hình RandomForest
    rf_result = rf_model.predict([features])[0]
    rf_probability = rf_model.predict_proba([features])[0]

    # Dự đoán từ mô hình SVM
    svm_result = svm_model.predict([features])[0]
    svm_probability = svm_model.predict_proba([features])[0]

    # Dự đoán từ mô hình XGBoost
    xgb_result = xgb_model.predict([features])[0]
    xgb_probability = xgb_model.predict_proba([features])[0]

    # Dự đoán từ mô hình Gradient Boosting
    gb_result = gb_model.predict([features])[0]
    gb_probability = gb_model.predict_proba([features])[0]

    # Dự đoán từ mô hình Logistic Regression
    lr_result = lr_model.predict([features])[0]
    lr_probability = lr_model.predict_proba([features])[0]

    # Dự đoán từ mô hình k-Nearest Neighbors (KNN)
    knn_result = knn_model.predict([features])[0]
    knn_probability = knn_model.predict_proba([features])[0]

    # Kết hợp dự đoán từ cả sáu mô hình
    ensemble_result = math.ceil((rf_result + svm_result + xgb_result + gb_result + lr_result + knn_result) / 6)
    ensemble_probability = [(rf_prob + svm_prob + xgb_prob + gb_prob + lr_prob + knn_prob) / 6 for rf_prob, svm_prob, xgb_prob, gb_prob, lr_prob, knn_prob in zip(rf_probability, svm_probability, xgb_probability, gb_probability, lr_probability, knn_probability)]

    return "Tài" if ensemble_result == 1 else "Xỉu", ensemble_probability

# Route mặc định, hiển thị trang nhập mã MD5
@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        md5 = request.form['md5']
        md5_results = load_data("datav4.txt")
        features = [extract_features(md5) for md5 in md5_results.keys()]
        labels = [1 if result == 'Tài' else 0 for result in md5_results.values()]
        rf_model = train_random_forest(features, labels)
        svm_model = train_svm(features, labels)
        xgb_model = train_xgboost(features, labels)
        gb_model = train_gradient_boosting(features, labels)
        lr_model = train_logistic_regression(features, labels)
        knn_model = train_knn(features, labels)
        predicted_result, _ = predict_result((rf_model, svm_model, xgb_model, gb_model, lr_model, knn_model), md5)
        return render_template('index.html', md5=md5, predicted_result=predicted_result)
    return render_template('index.html')

# Route để lưu kết quả thực tế vào dữ liệu
@app.route('/save', methods=['POST'])
def save():
    md5 = request.form['md5']
    actual_result = request.form['actual_result']
    md5_results = load_data("datav4.txt")
    md5_results[md5] = actual_result
    save_data("datav4.txt", md5_results)
    return redirect(url_for('index'))

if __name__ == "__main__":
    app.run(debug=True)
