#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# لضبط العشوائية للتكرار
np.random.seed(42)

# إعدادات حجم البيانات
num_customers = 300
num_products = 50
num_orders = 1000

# بيانات العملاء
cities = ['الرياض', 'جدة', 'الدمام', 'المدينة', 'أبها']
genders = ['ذكر', 'أنثى']

customers = pd.DataFrame({
    'customer_id': range(1, num_customers + 1),
    'gender': np.random.choice(genders, size=num_customers),
    'city': np.random.choice(cities, size=num_customers),
    'join_date': [datetime(2022, 1, 1) + timedelta(days=int(np.random.rand()*730)) for _ in range(num_customers)]
})

# بيانات المنتجات
categories = ['ملابس', 'إلكترونيات', 'أدوات منزلية', 'مكياج', 'كتب']

products = pd.DataFrame({
    'product_id': range(1, num_products + 1),
    'category': np.random.choice(categories, size=num_products),
    'price': np.random.randint(50, 1000, size=num_products),
    'on_sale': np.random.choice([True, False], size=num_products),
    'rating': np.round(np.random.uniform(2.5, 5.0, size=num_products), 1)
})

# بيانات الطلبات
orders = pd.DataFrame({
    'order_id': range(1, num_orders + 1),
    'order_date': [datetime(2023, 1, 1) + timedelta(days=int(np.random.rand()*580)) for _ in range(num_orders)],
    'customer_id': np.random.choice(customers['customer_id'], size=num_orders),
    'product_id': np.random.choice(products['product_id'], size=num_orders),
    'quantity': np.random.randint(1, 4, size=num_orders)
})

# إضافة سعر المنتج لكل طلب وحساب السعر الكلي
orders = orders.merge(products[['product_id', 'price']], on='product_id')
orders['total_price'] = orders['price'] * orders['quantity']

print("تم توليد البيانات بنجاح!")


# In[2]:


import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")
plt.rcParams['font.family'] = 'Arial'  # لضمان عرض الخطوط العربية بشكل جيد

# عرض أول 5 صفوف من الجداول
print("Customers Table:")
display(customers.head())

print("Products Table:")
display(products.head())

print("Orders Table:")
display(orders.head())

# استخراج الشهر من تاريخ الطلب
orders['order_month'] = orders['order_date'].dt.to_period('M').astype(str)

# المبيعات الشهرية
monthly_sales = orders.groupby('order_month')['total_price'].sum().reset_index()

plt.figure(figsize=(12, 6))
sns.lineplot(data=monthly_sales, x='order_month', y='total_price', marker='o', color='blue')
plt.title('Monthly Sales')
plt.xlabel('Month')
plt.ylabel('Total Sales (SAR)')
plt.xticks(rotation=45)
plt.grid(True)
plt.show()

# دمج الطلبات مع العملاء لأخذ المدينة
orders_with_city = orders.merge(customers[['customer_id', 'city']], on='customer_id')

city_sales = orders_with_city.groupby('city')['total_price'].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=city_sales.values, y=city_sales.index, palette='Blues_d')
plt.title('Sales by City')
plt.xlabel('Total Sales')
plt.ylabel('City')
plt.show()

# دمج الطلبات مع المنتجات لأخذ الفئة
orders_with_category = orders.merge(products[['product_id', 'category']], on='product_id')

category_sales = orders_with_category.groupby('category')['total_price'].sum().sort_values(ascending=False)

plt.figure(figsize=(8, 5))
sns.barplot(x=category_sales.values, y=category_sales.index, palette='viridis')
plt.title('Sales by Product Category')
plt.xlabel('Total Sales')
plt.ylabel('Category')
plt.show()

# أفضل 10 عملاء حسب الإنفاق
top_customers = orders.groupby('customer_id')['total_price'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
sns.barplot(x=top_customers.values, y=top_customers.index.astype(str), palette='magma')
plt.title('Top 10 Customers by Spending')
plt.xlabel('Total Spending')
plt.ylabel('Customer ID')
plt.show()


# In[3]:


import datetime as dt

# نحدد تاريخ اليوم الذي سنحسب منه recency (نفترض اليوم الأخير في البيانات)
reference_date = orders['order_date'].max() + pd.Timedelta(days=1)

# إنشاء جدول RFM لكل عميل
rfm = orders.groupby('customer_id').agg({
    'order_date': lambda x: (reference_date - x.max()).days,  # Recency: الفرق بالأيام
    'order_id': 'count',                                      # Frequency: عدد الطلبات
    'total_price': 'sum'                                      # Monetary: مجموع الإنفاق
}).reset_index()

# إعادة تسمية الأعمدة
rfm.columns = ['customer_id', 'Recency', 'Frequency', 'Monetary']

# عرض أول 5 صفوف
display(rfm.head())


# In[4]:


# تصنيف Recency: كل ما كان أقل (أحدث) يحصل على درجة أعلى (5 أفضل)
rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=range(5, 0, -1))

# تصنيف Frequency: كل ما كان أعلى يحصل على درجة أعلى (5 أفضل)
rfm['F_Score'] = pd.qcut(rfm['Frequency'].rank(method='first'), 5, labels=range(1, 6))

# تصنيف Monetary: كل ما كان أعلى يحصل على درجة أعلى (5 أفضل)
rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=range(1, 6))

# دمج الدرجات في عمود واحد لتسهيل التصنيف
rfm['RFM_Score'] = rfm['R_Score'].astype(str) + rfm['F_Score'].astype(str) + rfm['M_Score'].astype(str)

# عرض أول 10 عملاء مع تصنيف RFM
display(rfm.head(10))


# In[5]:


def rfm_segment(row):
    if row['R_Score'] >= 4 and row['F_Score'] >= 4 and row['M_Score'] >= 4:
        return 'Champions'
    elif row['R_Score'] >= 3 and row['F_Score'] >= 3 and row['M_Score'] >= 3:
        return 'Loyal Customers'
    elif row['R_Score'] >= 4 and row['F_Score'] <= 2:
        return 'Potential Loyalists'
    elif row['R_Score'] <= 2 and row['F_Score'] >= 4:
        return 'At Risk'
    elif row['R_Score'] <= 2 and row['F_Score'] <= 2:
        return 'Lost'
    else:
        return 'Others'

rfm['Segment'] = rfm.apply(rfm_segment, axis=1)

# عرض عدد العملاء في كل فئة
segment_counts = rfm['Segment'].value_counts()
print(segment_counts)

# رسم بياني لتوزيع الفئات
plt.figure(figsize=(8,5))
sns.barplot(x=segment_counts.index, y=segment_counts.values, palette='Set2')
plt.title('Customer Segments Distribution')
plt.xlabel('Segment')
plt.ylabel('Number of Customers')
plt.show()


# In[6]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# إعداد البيانات
X = rfm[['Frequency']]  # المتغير المستقل (عدد الطلبات)
y = rfm['Monetary']    # المتغير التابع (إجمالي الإنفاق)

# تقسيم البيانات (80% تدريب - 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء نموذج الانحدار الخطي
model = LinearRegression()
model.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار
y_pred = model.predict(X_test)

# تقييم النموذج
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse:.2f}')
print(f'R-squared: {r2:.2f}')

# رسم النتائج
plt.figure(figsize=(8,6))
plt.scatter(X_test, y_test, color='blue', label='Actual')
plt.scatter(X_test, y_pred, color='red', label='Predicted')
plt.title('Actual vs Predicted Monetary Spending')
plt.xlabel('Frequency (Number of Orders)')
plt.ylabel('Monetary (Total Spending)')
plt.legend()
plt.show()


# In[7]:


# المتغيرات المستقلة الجديدة
X = rfm[['Frequency', 'Recency']]  
y = rfm['Monetary']                

# تقسيم البيانات 80% تدريب و20% اختبار
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء النموذج
model = LinearRegression()
model.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار
y_pred = model.predict(X_test)

# تقييم النموذج
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Improved Model Mean Squared Error: {mse:.2f}')
print(f'Improved Model R-squared: {r2:.2f}')

# رسم التوقعات مقابل القيم الحقيقية
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred, color='purple')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Monetary Spending (Improved Model)')
plt.xlabel('Actual Spending')
plt.ylabel('Predicted Spending')
plt.show()


# In[8]:


from sklearn.ensemble import RandomForestRegressor

# إعداد المتغيرات المستقلة والمتغير التابع
X = rfm[['Frequency', 'Recency']]
y = rfm['Monetary']

# تقسيم البيانات (80% تدريب، 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# بناء نموذج الغابات العشوائية
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# التنبؤ على بيانات الاختبار
y_pred_rf = rf_model.predict(X_test)

# تقييم النموذج
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest MSE: {mse_rf:.2f}')
print(f'Random Forest R-squared: {r2_rf:.2f}')

# رسم القيم الفعلية مقابل المتوقعة
plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred_rf, color='green')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.title('Actual vs Predicted Monetary Spending (Random Forest)')
plt.xlabel('Actual Spending')
plt.ylabel('Predicted Spending')
plt.show()


# In[9]:


# استيراد المكتبات
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# تحضير البيانات
X = rfm[['Frequency', 'Recency']]
y = rfm['Monetary']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# نموذج الانحدار الخطي
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

# نموذج الغابات العشوائية
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

# طباعة النتائج
print("Linear Regression:")
print(f"  Mean Squared Error: {mse_lr:.2f}")
print(f"  R-squared: {r2_lr:.2f}\n")

print("Random Forest Regression:")
print(f"  Mean Squared Error: {mse_rf:.2f}")
print(f"  R-squared: {r2_rf:.2f}")

# رسم النتائج
plt.figure(figsize=(12,5))

plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred_lr, color='blue', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.title('Linear Regression')
plt.xlabel('Actual Spending')
plt.ylabel('Predicted Spending')

plt.subplot(1, 2, 2)
plt.scatter(y_test, y_pred_rf, color='green', alpha=0.6)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
plt.title('Random Forest Regression')
plt.xlabel('Actual Spending')
plt.ylabel('Predicted Spending')

plt.tight_layout()
plt.show()


# In[ ]:




