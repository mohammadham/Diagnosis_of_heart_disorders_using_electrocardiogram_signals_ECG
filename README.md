# Diagnosis_of_heart_disorders_using_electrocardiogram_signals_ECG
Diagnosis of heart disorders using electrocardiogram signals.
پروژه‌ی ساده برای تشخیص اختلال در قلب با استفاده از سیگنال‌های الکتروکاردیوگرام (ECG) 

توضیحی درباره کل پروژه:
این پروژه، به عنوان یک نمونه ساده از یادگیری عمیق برای تشخیص اختلال در قلب با استفاده از سیگنال‌های ECG، شامل ساخت یک شبکه عصبی ساده با استفاده از کتابخانه Tensorflow و آموزش آن بر روی داده‌های ECG است.
```
import numpy as np
import tensorflow as tf

# load the ECG dataset
ecg_data = np.load('ecg_data.npy')
ecg_labels = np.load('ecg_labels.npy')

# split the dataset into training and testing sets
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(ecg_data, ecg_labels, test_size=0.2, random_state=42)

# normalize the data
mean = x_train.mean()
std = x_train.std()
x_train = (x_train - mean) / std
x_test = (x_test - mean) / std

# define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=x_train[0].shape),
    tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# train the model
model.fit(x_train, y_train, batch_size=32, epochs=10, validation_data=(x_test, y_test))
```

شبکه عصبی استفاده شده:
```
# define the neural network architecture
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=x_train[0].shape),
    tf.keras.layers.Conv1D(32, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Conv1D(64, kernel_size=5, activation='relu'),
    tf.keras.layers.MaxPooling1D(pool_size=2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
```
در این نمونه، از یک شبکه عصبی ساده با دو لایه کانولوشن و دو لایه پرسپترون استفاده شده است که برای تشخیص اختلال در قلب با استفاده از سیگنال‌های ECG به کار گرفته می‌شود.خروجی این پروژه شامل دقت دسته‌بندی مدل بر روی داده‌های تست است که با استفاده از تابع evaluate در Tensorflow قابداع می‌شود. این دقت در اصطلاحات یادگیری ماشین به عنوان accuracy شناخته می‌شود و نشان می‌دهد که درصدی از داده‌های تست به درستی توسط مدل دسته‌بندی شده‌اند. به عنوان مثال، اگر دقت مدل ۹۵٪ باشد، این به این معنی است که ۹۵٪ از داده‌های تست به درستی توسط مدل دسته‌بندی شده‌اند.



