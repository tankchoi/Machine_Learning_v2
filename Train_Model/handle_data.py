import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Tải dữ liệu Auto MPG với tên cột đầy đủ
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"
column_names = ['mpg', 'cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin', 'car_name']
data = pd.read_csv(url, sep='\s+', names=column_names, na_values='?', comment='\t')
data['model_year'] = data['model_year'] + 1900
print("----------Kiểu dữ liệu----------")
data.info()
print("----------Số lượng giá trị duy nhất----------")
print(data.nunique().sort_values())
print("----------Tổng hợp thống kê mô tả----------")
print(data.describe())

# Biểu đồ phân phối của biến mục tiêu
plt.figure(figsize=[8, 4])
sns.histplot(data["mpg"], color='g', kde=True, bins=30, edgecolor="black", linewidth=2)
plt.title('Distribution of MPG Values in the Auto MPG Dataset')
plt.xlabel('Miles per Gallon (MPG)')
plt.ylabel('Frequency')
plt.show()

# Trực quan hóa các cột số
nf = data.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Vẽ boxplot cho các cột số
plt.figure(figsize=[15, 3 * math.ceil(len(nf) / 3)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf) / 3), 3, i + 1)
    data.boxplot(column=nf[i])
    plt.title(f'Boxplot of {nf[i]}', fontsize=10)
    plt.ylabel(nf[i])
plt.tight_layout()
plt.show()

# Vẽ biểu đồ phân phối cho các cột số
plt.figure(figsize=[15, 3 * math.ceil(len(nf) / 3)])
for i in range(len(nf)):
    plt.subplot(math.ceil(len(nf) / 3), 3, i + 1)
    sns.histplot(data[nf[i]], kde=True, bins=10, color=np.random.rand(3,), edgecolor="black", linewidth=2)
    plt.title(f'Distribution of {nf[i]}', fontsize=10)
    plt.xlabel(nf[i])
    plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Kiểm tra trùng lặp
rs, cs = data.shape
data.drop_duplicates(inplace=True)
new_rs, new_cs = data.shape
if (new_rs, new_cs) == (rs, cs):
    print('Bộ dữ liệu không có bất kỳ bản sao nào')
else:
    duplicates_removed = rs - new_rs
    print(f'Số lượng bản sao đã loại bỏ: {duplicates_removed}')


# Kiểm tra và thống kê các giá trị null
nvc = pd.DataFrame(data.isnull().sum().sort_values(),
columns=['Tổng số giá trị Null'])
nvc['Tỷ lệ phần trăm'] = round(nvc['Tổng số giá trị Null'] / data.shape[0], 3) * 100
print(nvc)


# Loại bỏ các giá trị ngoại lai
data1 = data.copy()
data3 = data.copy()

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

data1 = data1.reset_index(drop=True)
print(data1.head())
print('Trước khi loại bỏ các ngoại lệ, Bộ dữ liệu có {} mẫu.'.format(data3.shape[0]))
print('Sau khi loại bỏ các ngoại lệ, tập dữ liệu hiện có {} mẫu.'.format(data1.shape[0]))


# Vẽ heatmap cho ma trận tương quan
numeric_data = data.select_dtypes(include=['float64', 'int64'])
plt.figure(figsize=[8, 5])
sns.heatmap(numeric_data.corr(), annot=True, vmin=-1, vmax=1, center=0)
plt.title('Ma trận Tương Quan')
plt.show()




# Chuẩn hóa
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

data = data1
X = data[['cylinders', 'displacement', 'horsepower', 'weight', 'acceleration', 'model_year', 'origin']]
y = data['mpg']

# Chia dữ liệu thành tập train và test (85% train, 15% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42)

# Từ tập train, tách thêm tập validation (15% của tập data)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2142, random_state=42) 

# Khởi tạo và fit StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)

# Tính toán các giá trị thống kê
desc = X_train_scaled_df.describe().T

# Thêm các cột mean, std, min, max 
desc['mean'] = X_train_scaled_df.mean()
desc['std'] = X_train_scaled_df.std()
desc['min'] = X_train_scaled_df.min()
desc['25%'] = X_train_scaled_df.quantile(0.25)
desc['50%'] = X_train_scaled_df.median()
desc['75%'] = X_train_scaled_df.quantile(0.75)
desc['max'] = X_train_scaled_df.max()

print(desc)