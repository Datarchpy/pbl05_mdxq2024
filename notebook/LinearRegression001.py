import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.impute import SimpleImputer

# =======================================
# 1. データの読み込み
# =======================================
# 訓練用データの読み込み
base_train = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/train/base_train.csv')
processing_train = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/train/processing_train.csv')
actual_train = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/train/actual_train.csv')

# =======================================
# 2. データの概要確認と前処理
# =======================================
# 作業時間と合計時間の統計情報を表示
actual_train[['作業時間', '合計時間']].describe()

# 作業日を日付型に変換
actual_train['作業日'] = pd.to_datetime(actual_train['作業日'])

# 特定の機械のみを抽出（対象機種でフィルタリング）
actual_filtered = actual_train[actual_train['号機名'].isin(['グルアー', '2号機', '4号機', '6号機', '7号機', '8号機'])]

# 2020年2月4日以降のデータのみを使用
actual_filtered = actual_filtered[actual_filtered['作業日'] >= '2020-02-04']

# 作業時間と合計時間が正の値のデータのみを抽出（異常値除去）
actual_filtered = actual_filtered[(actual_filtered['作業時間'] > 0) & (actual_filtered['合計時間'] > 0)]

# =======================================
# 3. データ分布の可視化
# =======================================
# 作業時間のヒストグラム表示
plt.hist(actual_filtered['作業時間'], bins=100)
plt.title('作業時間の分布')
plt.show()

# 合計時間のヒストグラム表示
plt.hist(actual_filtered['合計時間'], bins=100)
plt.title('合計時間の分布')
plt.show()

# =======================================
# 4. 訓練データの準備
# =======================================
# processing_trainとactual_trainを結合（受注番号と号機名をキーとして）
train_merged = pd.merge(processing_train, actual_train, on=['受注番号', '号機名'])

# 2020年2月4日以降のデータに限定
train_merged = train_merged[train_merged['作業日'] >= '2020-02-04']

# 必要な列のみを抽出
train_data = train_merged[['受注番号', '号機名', '作業日', '数量1','数量2','数量3',
                           '作業時間', '合計時間']]

# 特定の機械のみを対象とする
train_data = train_data[train_data['号機名'].isin(['グルアー', '2号機', '4号機', '6号機', '7号機', '8号機'])]

# 作業時間と合計時間が正の値のデータのみを抽出
train_data = train_data[(train_data['作業時間'] > 0) & (train_data['合計時間'] > 0)]

# =======================================
# 5. データ型の変換と欠損値処理
# =======================================
# 数値型に変換（エラーの場合はNaNとする）
train_data['数量1'] = pd.to_numeric(train_data['数量1'], errors='coerce')
train_data['数量2'] = pd.to_numeric(train_data['数量2'], errors='coerce')
train_data['数量3'] = pd.to_numeric(train_data['数量3'], errors='coerce')
train_data['作業時間'] = pd.to_numeric(train_data['作業時間'], errors='coerce')
train_data['合計時間'] = pd.to_numeric(train_data['合計時間'], errors='coerce')

# 欠損値を含む行を削除
train_data = train_data.dropna(subset=['数量1','数量2', '数量3', '作業時間', '合計時間'])

# =======================================
# 6. 対数変換（データの正規化）
# =======================================
# 各数値列に対して対数変換を適用（log1p = log(1+x)を使用してゼロ値に対応）
train_data['log_数量1'] = np.log1p(train_data['数量1'])
train_data['log_数量2'] = np.log1p(train_data['数量2'])
train_data['log_数量3'] = np.log1p(train_data['数量3'])
train_data['log_作業時間'] = np.log1p(train_data['作業時間'])
train_data['log_合計時間'] = np.log1p(train_data['合計時間'])

# =======================================
# 7. 訓練・検証データの分割
# =======================================
# 2020年6月1日より前を訓練データとする
train_all = train_data[train_data['作業日'] < '2020-06-01']

# 2020年6月1日以降を検証データとする
val_all = train_data[train_data['作業日'] >= '2020-06-01']

# =======================================
# 8. 線形回帰モデルの学習
# =======================================
# 線形回帰モデルのインスタンス作成
regr = linear_model.LinearRegression()

# 特徴量（対数変換後の数量1,2,3）で目的変数（対数変換後の作業時間、合計時間）を予測するモデルを学習
regr.fit(train_all[['log_数量1', 'log_数量2', 'log_数量3']], 
         train_all[['log_作業時間', 'log_合計時間']])

# =======================================
# 9. 検証データでの予測と評価
# =======================================
# 検証データに対する予測（対数スケール）
y_hat_log = regr.predict(val_all[['log_数量1', 'log_数量2', 'log_数量3']])

# 対数変換を元に戻す（expm1 = exp(x)-1）
y_hat = np.expm1(y_hat_log)

# MAE（平均絶対誤差）の計算
print('正味作業時間のMAE:', np.abs(val_all['作業時間'] - y_hat[:, 0]).mean())
print('付帯作業時間のMAE:', np.abs(val_all['合計時間'] - y_hat[:, 1]).mean())

# =======================================
# 10. 予測結果の可視化
# =======================================
# 正味作業時間の予測vs実測値の散布図
fig, ax = plt.subplots(figsize=(4, 4), dpi=90)
ax.set_xlabel('prediction', fontsize=10)
ax.set_ylabel('ground truth', fontsize=10)
# 理想的な予測線（y=x）を赤線で表示
ax.plot([i for i in range(int(val_all['作業時間'].max()))], 
        [i for i in range(int(val_all['作業時間'].max()))], color='red')
# 実際の予測値vs実測値の散布図
ax.scatter(y_hat[:, 0], val_all['作業時間'])
ax.grid()
ax.set_aspect('equal')
plt.title('正味作業時間：予測vs実測')
plt.show()

# 付帯作業時間の予測vs実測値の散布図
fig, ax = plt.subplots(figsize=(4, 4), dpi=90)
ax.set_xlabel('prediction', fontsize=10)
ax.set_ylabel('ground truth', fontsize=10)
# 理想的な予測線（y=x）を赤線で表示
ax.plot([i for i in range(int(val_all['合計時間'].max()))], 
        [i for i in range(int(val_all['合計時間'].max()))], color='red')
# 実際の予測値vs実測値の散布図
ax.scatter(y_hat[:, 1], val_all['合計時間'])
ax.grid()
ax.set_aspect('equal')
plt.title('付帯作業時間：予測vs実測')
plt.show()

# =======================================
# 11. テストデータの読み込みと前処理
# =======================================
# テストデータの読み込み
processing_test = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/test/processing_test.csv')
actual_test = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/test/actual_test.csv')

# テストデータを結合
test_merged = pd.merge(actual_test, processing_test)

# 必要な列のみを抽出
test_data = test_merged[['index', '受注番号', '号機名', '数量1', '数量2', '数量3']]

# データのコピーを作成（元データを保持）
test_data = test_data.copy()

# =======================================
# 12. テストデータの欠損値処理
# =======================================
# 欠損値の確認
print("テストデータの欠損値数:")
print(test_merged[['数量1', '数量2', '数量3']].isnull().sum())

# 欠損値を中央値で補完（より安定した補完方法）
test_data['数量1'] = test_data['数量1'].fillna(test_data['数量1'].median())
test_data['数量2'] = test_data['数量2'].fillna(test_data['数量2'].median())
test_data['数量3'] = test_data['数量3'].fillna(test_data['数量3'].median())

# テストデータの対数変換
test_data['log_数量1'] = np.log1p(test_data['数量1'])
test_data['log_数量2'] = np.log1p(test_data['数量2'])
test_data['log_数量3'] = np.log1p(test_data['数量3'])

# =======================================
# 13. テストデータの予測
# =======================================
# 学習済みモデルでテストデータの予測（対数スケール）
y_hat_test_log = regr.predict(test_data[['log_数量1', 'log_数量2', 'log_数量3']])

# 対数変換を元に戻す
y_hat_test = np.expm1(y_hat_test_log)

# =======================================
# 14. 提出用ファイルの作成
# =======================================
# 予測結果をDataFrameに格納
submit = pd.DataFrame({
    'index': test_data['index'], 
    '正味作業時間': y_hat_test[:, 0], 
    '付帯作業時間': y_hat_test[:, 1]
})

# CSVファイルとして保存（ヘッダーとインデックスなし）
submit.to_csv('s003submission.csv', index=False, header=False)

print("予測完了！提出用ファイル's003submission.csv'が作成されました。")
