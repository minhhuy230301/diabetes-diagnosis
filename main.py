from flask_cors import CORS, cross_origin # thêm thư viện này để giải quyết vấn đề CORS
from flask import Flask,request,jsonify # thêm thư viện này để tạo API
import pandas as pd # Xử lý và phân tích dữ liệu
import numpy as np # Thực hiện các phép toán số học
import seaborn as sns # Vẽ biểu đồ
import matplotlib.pyplot as plt # Vẽ đồ thị
plt.show()
from matplotlib import style # Thay đổi style của đồ thị
from sklearn.preprocessing import  StandardScaler # Chuẩn hóa dữ liệu
from sklearn.model_selection import train_test_split # Chia dữ liệu thành 2 phần train và test
from sklearn.linear_model import LogisticRegression # Mô hình hồi quy logistic
from sklearn import svm # Mô hình máy vector hỗ trợ
from sklearn.metrics import accuracy_score,classification_report # Đánh giá mô hình
import warnings # Bỏ qua các cảnh báo
warnings.filterwarnings('ignore') # Bỏ qua các cảnh báo

app = Flask(__name__) # Khởi tạo ứng dụng Flask
CORS(app, resources={r"/process_data": {"origins": "http://localhost:5173"}}) # Giải quyết vấn đề CORS

# @app.route('/', method=['GET'])
# def hello():
#     return 'Hello'
@app.route('/process_data',methods=['OPTION','POST'])
@cross_origin(origin='http://localhost:5173', headers=['Content-Type'])
def process_data():
    # nhận dữ liệu từ front_end gửi lên
    data = request.get_json() # nhận dữ liệu từ front_end gửi lên
    # Trích xuất giá trị số thực từ dictionary
    input_data = [float(value) for value in data.values()]

    # bắt đầu phân tích sử dụng phương pháp hồi quy tuyến tính
    df = pd.read_csv('diabetes.csv')
    print(df.head()) # Hiển thị 5 dòng đầu tiên của dữ liệu

    # print(df.shape)
    # print(df.describe())
    # sns.countplot(x='Outcome',data=df)
    print(df.corr()['Outcome'].sort_values()) # Hiển thị tương quan giữa các cột với cột Outcome
    x = df.drop(columns='Outcome', axis=1) # Lấy tất cả gán vào biến x các cột ngoại trừ cột Outcome
    y = df['Outcome'] # Lấy cột Outcome gán vào biến y
    scaler = StandardScaler() # Khởi tạo đối tượng chuẩn hóa dữ liệu
    scaler.fit(x) # Huấn luyện bộ chuân hóa dữ liệu x
    standardized_data = scaler.transform(x) # Chuẩn hóa dữ liệu x
    x = standardized_data # Gán dữ liệu chuẩn hóa vào x
    y = df['Outcome'] # Gán cột Outcome vào y
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42) # Chia dữ liệu thành 2 phần train và test với 20% dữ liệu test
    print(x_train.shape) # Hiển thị số dòng và số cột của dữ liệu train
    print(x_test.shape) # Hiển thị số dòng và số cột của dữ liệu test
    print(y_train.shape) # Hiển thị số dòng của dữ liệu train
    print(y_test.shape) # Hiển thị số dòng của dữ liệu test
    logreg = LogisticRegression() # Khởi tạo mô hình hồi quy logistic
    logreg.fit(x_train, y_train) # Huấn luyện mô hình hồi quy logistic
    logreg_pred = logreg.predict(x_test)   # Dự đoán dữ liệu test
    logreg_acc = accuracy_score(logreg_pred, y_test) # Tính độ chính xác mô hình
    print("test Accuracy:{:.2f}%".format(logreg_acc * 100)) # Hiển thị độ chính xác mô hình (75.32)
    print(classification_report(y_test, logreg_pred)) # Hiển thị báo cáo đánh giá mô hình
    # input_data = (5,180,72,22,160,33.6,0.672,50)
    # input_data = (1, 85,66 ,29 , 0 ,26.6 ,0.351 ,31 )
    input_data_as_numpy_array = np.asarray(input_data) # Chuyển dữ liệu mới thành mảng numpy
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1) # Reshape dữ liệu mới
    scalar_data = scaler.transform(input_data_reshaped) # Chuẩn hóa dữ liệu mới
    prediction = logreg.predict(scalar_data) # Dự đoán dữ liệu mới
    result_list = prediction.tolist() # Chuyển kết quả dự đoán thành list
    return jsonify(result_list) # Trả về kết quả dự đoán dưới dạng json

@app.route('/about/<username>')
def about_page(username):
    return f'This is the about page of {username}'



if __name__ == '__main__':
    app.run(debug=True)


