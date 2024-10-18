import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler  # Nhập StandardScaler
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Tải dữ liệu Auto MPG với tên cột đầy đủ
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
data = pd.read_csv(url, sep='\s+', names=column_names, na_values='?', comment='\t')

# Xử lý dữ liệu thiếu
data = data.dropna()
# Loại bỏ các giá trị ngoại lai
data1 = data.copy()

# Xác định các đặc trưng số
features1 = data.select_dtypes(include=['float64', 'int64']).columns.tolist()
# Loại bỏ ngoại lệ dựa trên IQR
for i in features1:
    Q1 = data1[i].quantile(0.1)
    Q3 = data1[i].quantile(0.9)
    IQR = Q3 - Q1
    
    # Lọc các giá trị không vượt quá ngưỡng trên
    data1 = data1[data1[i] <= (Q3 + (1.5 * IQR))]
    # Lọc các giá trị không vượt quá ngưỡng dưới
    data1 = data1[data1[i] >= (Q1 - (1.5 * IQR))]

# Đặt lại chỉ số cho DataFrame sau khi lọc
data1 = data1.reset_index(drop=True)
data = data1
# Chuyển đổi 'model_year' từ 2 chữ số thành 4 chữ số
data['model_year'] = data['model_year'] + 1900

# Chọn thuộc tính đầu vào và biến mục tiêu
X = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']]
y = data['mpg']

# Chia dữ liệu thành tập train và test (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Từ tập train, tách thêm tập validation (15% của tập data)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2142, random_state=42) 

# Khởi tạo StandardScaler và chuẩn hóa dữ liệu
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Chỉ sử dụng fit_transform cho tập huấn luyện
X_valid_scaled = scaler.transform(X_valid)        # Chỉ sử dụng transform cho tập xác thực
X_test_scaled = scaler.transform(X_test)          # Chỉ sử dụng transform cho tập kiểm tra

# Thiết lập phạm vi giá trị alpha cho GridSearchCV
alpha_values = np.logspace(-4, 4, 50)  # Các giá trị alpha từ 10^-4 đến 10^4

# Thiết lập GridSearchCV để tìm giá trị alpha tốt nhất
param_grid = {'alpha': alpha_values}
grid_search = GridSearchCV(Ridge(), param_grid, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

# Lấy giá trị alpha tốt nhất từ GridSearchCV
best_alpha = grid_search.best_params_['alpha']
print(f"Giá trị alpha tốt nhất: {best_alpha}")

# Huấn luyện mô hình Ridge Regression với alpha tốt nhất
model = Ridge(alpha=best_alpha)
model.fit(X_train_scaled, y_train)

# Dự đoán trên tập train, valid và test
y_train_pred = model.predict(X_train_scaled)
y_valid_pred = model.predict(X_valid_scaled)
y_test_pred = model.predict(X_test_scaled)

# Tính toán lỗi
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
valid_rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))

train_mae = mean_absolute_error(y_train, y_train_pred)
valid_mae = mean_absolute_error(y_valid, y_valid_pred)
test_mae = mean_absolute_error(y_test, y_test_pred)

# Tính toán R^2
train_r2 = r2_score(y_train, y_train_pred)
valid_r2 = r2_score(y_valid, y_valid_pred)
test_r2 = r2_score(y_test, y_test_pred)

# Tạo DataFrame để theo dõi quá trình
results = pd.DataFrame({
    'Dataset': ['Train', 'Validation', 'Test'],
    'RMSE': [train_rmse, valid_rmse, test_rmse],
    'MAE': [train_mae, valid_mae, test_mae],
    'R^2': [train_r2, valid_r2, test_r2]
})

print("Kết quả lỗi, MAE và R^2:")
print(results)

# Dự đoán cho mẫu dữ liệu
sample_input = np.array([[8, 307.0, 130.0, 3504.0, 12.0, 1970, 1]])  # Replace with appropriate values
sample_input_scaled = scaler.transform(sample_input)  # Chuẩn hóa mẫu dữ liệu

# Dự đoán với mô hình đã huấn luyện
sample_prediction = model.predict(sample_input_scaled)

print("Dự đoán cho mẫu dữ liệu:", sample_prediction)

# Biểu đồ dự đoán so với giá trị thực
plt.figure(figsize=(18, 6))

# Biểu đồ dự đoán so với giá trị thực cho dữ liệu train
plt.subplot(1, 3, 1)
plt.scatter(y_train, y_train_pred, alpha=0.5)
plt.plot([min(y_train), max(y_train)], [min(y_train), max(y_train)], color='red', linestyle='--')
plt.xlabel('Giá trị thực (Train)')
plt.ylabel('Dự đoán (Train)')
plt.title('Dữ liệu Train')

# Biểu đồ dự đoán so với giá trị thực cho dữ liệu validation
plt.subplot(1, 3, 2)
plt.scatter(y_valid, y_valid_pred, alpha=0.5)
plt.plot([min(y_valid), max(y_valid)], [min(y_valid), max(y_valid)], color='red', linestyle='--')
plt.xlabel('Giá trị thực (Validation)')
plt.ylabel('Dự đoán (Validation)')
plt.title('Dữ liệu Validation')

# Biểu đồ dự đoán so với giá trị thực cho dữ liệu test
plt.subplot(1, 3, 3)
plt.scatter(y_test, y_test_pred, alpha=0.5)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linestyle='--')
plt.xlabel('Giá trị thực (Test)')
plt.ylabel('Dự đoán (Test)')
plt.title('Dữ liệu Test')

plt.tight_layout()
plt.show()

# Lưu mô hình vào file
model_filename = r'E:\MachineLearning\BE\ridge_regression_model.pkl'
joblib.dump(model, model_filename)
