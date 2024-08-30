import pandas as pd # Xử lý và phân tích dữ liệu
import numpy as np # Thực hiện các phép toán số học
import seaborn # Vẽ biểu đồ
import seaborn as sns # Vẽ biểu đồ
import matplotlib.pyplot as plt # Vẽ đồ thi
plt.show()
from matplotlib import style
from sklearn.preprocessing import  StandardScaler # Chuẩn hóa dữ liệu
from sklearn.model_selection import train_test_split # Chia dữ liệu thành 2 phần train và test
from sklearn.linear_model import LogisticRegression # Mô hình hồi quy logistic
from sklearn import svm # Mô hình máy vector hỗ trợ
from sklearn.metrics import accuracy_score,classification_report # Đánh giá mô hình
import warnings # Bỏ qua các cảnh báo
warnings.filterwarnings('ignore')

df = pd.read_csv('diabetes.csv') # Đọc dữ liệu từ file csv
print(df.head()) # Hiển thị 5 dòng đầu tiên của dữ liệu
print(df.shape) # Hiển thị số dòng và số cột của dữ liệu
print(df.describe()) # Hiển thị thông tin thống kê của dữ liệu
# sns.countplot(x='Outcome',data=df)

x=df.drop(columns='Outcome',axis=1) # Lấy tất cả gán vào biến x các cột ngoại trừ cột Outcome
y=df['Outcome'] # Lấy cột Outcome gán vào biến y
scaler = StandardScaler() # Khởi tạo đối tượng chuẩn hóa dữ liệu
scaler.fit(x) # Huấn luyện bộ chuân hóa dữ liệu x
standardized_data=scaler.transform(x) # Chuẩn hóa dữ liệu x
x=standardized_data # Gán dữ liệu chuẩn hóa vào x
y=df['Outcome'] # Gán cột Outcome vào y
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42) # Chia dữ liệu thành 2 phần train và test với 20% dữ liệu test
print(x_train.shape) # Hiển thị số dòng và số cột của dữ liệu train
print(x_test.shape) # Hiển thị số dòng và số cột của dữ liệu test
print(y_train.shape) # Hiển thị số dòng của dữ liệu train
print(y_test.shape) # Hiển thị số dòng của dữ liệu test
logreg = LogisticRegression() # Khởi tạo mô hình hồi quy logistic
logreg.fit(x_train,y_train) # Huấn luyện mô hình hồi quy logistic
logreg_pred = logreg.predict(x_test)    # Dự đoán dữ liệu test
logreg_acc = accuracy_score(logreg_pred,y_test) # Tính độ chính xác mô hình
print("test Accuracy:{:.2f}%".format(logreg_acc*100)) # Hiển thị độ chính xác mô hình (75.32)
print(classification_report(y_test,logreg_pred)) # Hiển thị báo cáo đánh giá mô hình
# input_data = (5,180,72,22,160,33.6,0.672,50)
# Dự đoán với dữ liệu mới
input_data = (8,183,64,0,0,23.3,0.672,32) # Dữ liệu mới
input_data_as_numpy_array = np.asarray(input_data) # Chuyển dữ liệu mới thành mảng numpy
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1) # Reshape dữ liệu mới
scalar_data = scaler.transform(input_data_reshaped) # Chuẩn hóa dữ liệu mới
prediction = logreg.predict(scalar_data) # Dự đoán dữ liệu mới
print(prediction) # Hiển thị kết quả dự đoán
 
