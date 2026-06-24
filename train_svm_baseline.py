import os
import re
import jieba
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, learning_curve
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


"""
Baseline 实验：TF-IDF + SVM 中文商品评论情感分类
实验流程：
1. 读取 online_shopping_10_cats 数据集
2. 使用全部商品类别
3. 文本清洗
4. jieba 中文分词
5. 划分训练集和测试集
6. TF-IDF 特征提取
7. LinearSVC 模型训练
8. 模型评估
9. 结果可视化
10. 保存模型和实验结果
"""

# =========================
# 0. 基础配置
# =========================

data_path = '/data2/nfs_node1/gb/gb_project/Project/EmotionAnalysis/datasets/online_shopping_10_cats.csv'

save_dir = "svm_weight/baseline_all_cats"
fig_dir = os.path.join(save_dir, "figures")

os.makedirs(save_dir, exist_ok=True)
os.makedirs(fig_dir, exist_ok=True)

# 图表中全部使用英文，避免中文乱码
plt.rcParams['axes.unicode_minus'] = False


# =========================
# 1. 类别英文映射
# =========================

cat_name_map = {
    '书籍': 'Books',
    '平板': 'Tablet',
    '手机': 'Phone',
    '水果': 'Fruit',
    '洗发水': 'Shampoo',
    '热水器': 'Water Heater',
    '蒙牛': 'Mengniu',
    '衣服': 'Clothes',
    '计算机': 'Computer',
    '酒店': 'Hotel'
}


# =========================
# 2. 读取数据
# =========================

pd_all = pd.read_csv(data_path)

print("原始数据列名：")
print(pd_all.columns)

# 使用全部类别
pd_data = pd_all.copy()

# 去除空值
pd_data = pd_data.dropna(subset=['review', 'label', 'cat'])

# 标签转为整数
pd_data['label'] = pd_data['label'].astype(int)

# 增加英文类别列，专门用于画图
pd_data['cat_en'] = pd_data['cat'].map(cat_name_map)

# 如果有没映射到的类别，保留原名
pd_data['cat_en'] = pd_data['cat_en'].fillna(pd_data['cat'])

print("\n========== 数据集基本信息 ==========")
print('评论数目（总体）：%d' % pd_data.shape[0])
print('评论数目（正向）：%d' % pd_data[pd_data.label == 1].shape[0])
print('评论数目（负向）：%d' % pd_data[pd_data.label == 0].shape[0])

print("\n各类别样本分布：")
print(pd_data['cat'].value_counts())

print("\n各类别正负样本分布：")
print(pd_data.groupby(['cat', 'label']).size())


# =========================
# 3. 保存数据统计表
# =========================

pd_data['cat'].value_counts().to_csv(
    os.path.join(save_dir, "category_distribution.csv"),
    encoding="utf-8-sig"
)

pd_data.groupby(['cat', 'label']).size().to_csv(
    os.path.join(save_dir, "category_label_distribution.csv"),
    encoding="utf-8-sig"
)


# =========================
# 4. 数据分布可视化
# =========================

# 4.1 正负样本分布图
label_counts = pd_data['label'].value_counts().sort_index()

plt.figure(figsize=(6, 4))
plt.bar(['Negative', 'Positive'], label_counts.values)
plt.title('Label Distribution')
plt.xlabel('Sentiment Label')
plt.ylabel('Number of Reviews')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'label_distribution.png'), dpi=300)
plt.close()


# 4.2 商品类别分布图，类别名使用英文
cat_counts = pd_data['cat_en'].value_counts()

plt.figure(figsize=(11, 5))
plt.bar(cat_counts.index, cat_counts.values)
plt.title('Product Category Distribution')
plt.xlabel('Product Category')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'category_distribution.png'), dpi=300)
plt.close()


# 4.3 各类别正负样本堆叠图
cat_label_counts = pd_data.groupby(['cat_en', 'label']).size().unstack(fill_value=0)

# 保证列顺序是 0, 1
if 0 not in cat_label_counts.columns:
    cat_label_counts[0] = 0
if 1 not in cat_label_counts.columns:
    cat_label_counts[1] = 0

cat_label_counts = cat_label_counts[[0, 1]]
cat_label_counts.columns = ['Negative', 'Positive']

plt.figure(figsize=(11, 5))
cat_label_counts.plot(kind='bar', figsize=(11, 5))
plt.title('Category Sentiment Distribution')
plt.xlabel('Product Category')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'category_label_distribution.png'), dpi=300)
plt.close()


# =========================
# 5. 文本清洗
# =========================

def clean_text(text):
    """
    文本清洗：
    只保留中文、英文和数字，其余字符替换为空格
    """
    text = str(text)
    text = re.sub(r'[^\u4e00-\u9fa5a-zA-Z0-9]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# =========================
# 6. jieba 分词
# =========================

def cut_words(text):
    """
    中文分词：
    先清洗文本，再使用 jieba 分词
    """
    text = clean_text(text)
    words = jieba.lcut(text)
    return ' '.join(words)


print("\n正在进行 jieba 分词...")
pd_data['cut_review'] = pd_data['review'].apply(cut_words)
print("分词完成。")


# =========================
# 7. 准备输入和标签
# =========================

X = pd_data['cut_review']
y = pd_data['label']


# =========================
# 8. 划分训练集和测试集
# =========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\n========== 数据划分 ==========")
print("训练集数量：", len(X_train))
print("测试集数量：", len(X_test))


# =========================
# 9. TF-IDF 特征提取
# =========================

tfidf = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

print("\n正在提取 TF-IDF 特征...")

X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("TF-IDF 特征提取完成。")
print("训练集特征维度：", X_train_tfidf.shape)
print("测试集特征维度：", X_test_tfidf.shape)


# =========================
# 10. SVM 模型训练
# =========================

print("\n正在训练 LinearSVC 模型...")

model = LinearSVC(
    C=1.0,
    max_iter=5000,
    random_state=42
)

model.fit(X_train_tfidf, y_train)

print("模型训练完成。")


# =========================
# 11. 模型预测
# =========================

y_pred = model.predict(X_test_tfidf)


# =========================
# 12. 模型评估
# =========================

acc = accuracy_score(y_test, y_pred)
report_text = classification_report(y_test, y_pred, digits=4)
report_dict = classification_report(y_test, y_pred, output_dict=True)
cm = confusion_matrix(y_test, y_pred)

print("\n========== Baseline 实验结果 ==========")
print("准确率 Accuracy：%.4f" % acc)

print("\n分类报告：")
print(report_text)

print("\n混淆矩阵：")
print(cm)


# =========================
# 13. 保存实验结果
# =========================

report_df = pd.DataFrame(report_dict).transpose()

report_df.to_csv(
    os.path.join(save_dir, "classification_report.csv"),
    encoding="utf-8-sig"
)

with open(os.path.join(save_dir, "result.txt"), "w", encoding="utf-8") as f:
    f.write("Baseline: TF-IDF + LinearSVC\n")
    f.write("Dataset: online_shopping_10_cats, all categories\n\n")
    f.write("Total samples: %d\n" % pd_data.shape[0])
    f.write("Train samples: %d\n" % len(X_train))
    f.write("Test samples: %d\n" % len(X_test))
    f.write("Accuracy: %.4f\n\n" % acc)

    f.write("Classification Report:\n")
    f.write(report_text)

    f.write("\nConfusion Matrix:\n")
    f.write(str(cm))


# =========================
# 14. 混淆矩阵可视化
# =========================

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['Negative', 'Positive']
)

disp.plot()
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'confusion_matrix.png'), dpi=300)
plt.close()


# =========================
# 15. Precision / Recall / F1 可视化
# =========================

metric_df = report_df.loc[['0', '1'], ['precision', 'recall', 'f1-score']]
metric_df.index = ['Negative', 'Positive']

plt.figure(figsize=(7, 5))
metric_df.plot(kind='bar')
plt.title('Precision / Recall / F1-score')
plt.xlabel('Class')
plt.ylabel('Score')
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.legend(loc='lower right')
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'classification_metrics.png'), dpi=300)
plt.close()


# =========================
# 16. Learning Curve
# =========================

print("\n正在绘制 Learning Curve，这一步可能会稍慢...")

train_sizes, train_scores, val_scores = learning_curve(
    estimator=LinearSVC(
        C=1.0,
        max_iter=5000,
        random_state=42
    ),
    X=X_train_tfidf,
    y=y_train,
    train_sizes=np.linspace(0.2, 1.0, 5),
    cv=3,
    scoring='accuracy',
    n_jobs=-1
)

train_scores_mean = train_scores.mean(axis=1)
val_scores_mean = val_scores.mean(axis=1)

plt.figure(figsize=(7, 5))
plt.plot(train_sizes, train_scores_mean, marker='o', label='Training Accuracy')
plt.plot(train_sizes, val_scores_mean, marker='o', label='Validation Accuracy')
plt.title('Learning Curve')
plt.xlabel('Number of Training Samples')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(fig_dir, 'learning_curve.png'), dpi=300)
plt.close()


# =========================
# 17. 保存模型和 TF-IDF
# =========================

model_path = os.path.join(save_dir, 'svm_sentiment_model.pkl')
tfidf_path = os.path.join(save_dir, 'tfidf_vectorizer.pkl')

joblib.dump(model, model_path)
joblib.dump(tfidf, tfidf_path)


# =========================
# 18. 保存测试集预测结果
# =========================

test_result_df = pd.DataFrame({
    "text_cut": X_test.values,
    "true_label": y_test.values,
    "pred_label": y_pred
})

test_result_df.to_csv(
    os.path.join(save_dir, "test_predictions.csv"),
    index=False,
    encoding="utf-8-sig"
)


# =========================
# 19. 结束提示
# =========================

print("\n========== 文件保存完成 ==========")
print("模型已保存：", model_path)
print("TF-IDF 已保存：", tfidf_path)
print("实验结果已保存：", os.path.join(save_dir, "result.txt"))
print("分类报告已保存：", os.path.join(save_dir, "classification_report.csv"))
print("预测结果已保存：", os.path.join(save_dir, "test_predictions.csv"))
print("类别分布表已保存：", os.path.join(save_dir, "category_distribution.csv"))
print("类别正负分布表已保存：", os.path.join(save_dir, "category_label_distribution.csv"))
print("可视化图片已保存到：", fig_dir)