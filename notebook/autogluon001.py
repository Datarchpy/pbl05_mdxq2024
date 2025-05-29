# AutoGluonライブラリのインストール
! pip install autogluon

import pandas as pd
from autogluon.tabular import TabularPredictor

# =======================================
# 1. 訓練データの読み込み
# =======================================
# 各種訓練データファイルを読み込み
base_train = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/train/base_train.csv')
processing_train = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/train/processing_train.csv')
actual_train = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/train/actual_train.csv')

# =======================================
# 2. 実績データの前処理
# =======================================
# 作業日を日付型に変換（エラーが発生した場合はNaNとする）
actual_train['作業日'] = pd.to_datetime(actual_train['作業日'], errors='coerce')

# 所要時間に欠損値がある行を削除してインデックスをリセット
actual_train = actual_train.dropna(subset=['所要時間']).reset_index(drop=True)

# 付帯作業時間の計算
# 2020年2月4日以降：合計時間を使用
# それより前：所要時間から作業時間と残業時間を差し引いた値を使用
actual_train['付帯作業時間'] = actual_train.apply(
    lambda row: row['合計時間'] if row['作業日'] >= pd.Timestamp('2020-02-04') 
    else row['所要時間'] - (row['作業時間'] + row['残業時間']),
    axis=1
)

# 付帯作業時間が負の値のデータを除外（データ品質管理）
actual_train = actual_train[actual_train['付帯作業時間'] >= 0].reset_index(drop=True)

# =======================================
# 3. 訓練データの結合と前処理
# =======================================
# base_trainとprocessing_trainを受注番号をキーとして結合
train_data = pd.merge(base_train, processing_train, on='受注番号', how='left')

# actual_trainの必要な列のみを結合
train_data = pd.merge(
    train_data, 
    actual_train[['受注番号', '号機名', '作業時間', '付帯作業時間']], 
    on=['受注番号', '号機名'], 
    how='left'
)

# 対象機械のみを抽出
train_data = train_data[train_data['号機名'].isin(['グルアー', '2号機', '4号機', '6号機', '7号機', '8号機'])]

# 異常値の除去（作業時間と付帯作業時間が0以上150以下のデータのみ）
train_data = train_data[
    (train_data['作業時間'] >= 0) & (train_data['作業時間'] <= 150) &
    (train_data['付帯作業時間'] >= 0) & (train_data['付帯作業時間'] <= 150)
]

# =======================================
# 4. 欠損値の多い列の除去
# =======================================
# 50%以上欠損している列を特定するための閾値
threshold = 0.5

# 各列の欠損率を計算
missing_rate_train = train_data.isnull().mean()

# 50%以上欠損している列を取得
columns_to_drop = missing_rate_train[missing_rate_train > threshold].index

# 該当する列を削除
train_data = train_data.drop(columns=columns_to_drop, axis=1)

print(f"削除された列: {list(columns_to_drop)}")

# =======================================
# 5. テストデータの読み込みと処理
# =======================================
# テストデータファイルを読み込み
base_test = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/test/base_test.csv')
processing_test = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/test/processing_test.csv')
actual_test = pd.read_csv('/kaggle/input/pbl05-data/3-2_PBL05_data/test/actual_test.csv')

# テストデータを結合
test_data = pd.merge(base_test, processing_test, on='受注番号', how='left')
test_data = pd.merge(
    test_data, 
    actual_test[['index', '受注番号', '号機名']], 
    on=['受注番号', '号機名'], 
    how='left'
)

# indexでソートしてインデックスをリセット
test_data = test_data.sort_values(by='index').reset_index(drop=True)

# =======================================
# 6. 特徴量の準備
# =======================================
# 学習データとテストデータの共通列を抽出
common_features = [col for col in train_data.columns if col in test_data.columns]

# 特徴量リストを設定
features = common_features

print(f"共通特徴量の数: {len(features)}")
print(f"共通特徴量: {features}")

# 予測に使用しない列を指定
exclude_columns = ['受注番号', '号機名', '作業時間', '付帯作業時間']

# 最終的な特徴量リストを作成（除外列を除く）
features = [col for col in train_data.columns if col not in exclude_columns]

# 列名の重複確認
duplicates = train_data.columns[train_data.columns.duplicated()]
print(f"重複している列名: {duplicates.tolist()}")

# =======================================
# 7. AutoGluonモデルの学習（作業時間）
# =======================================
print("\n=== 作業時間予測モデルの学習開始 ===")

# 作業時間予測用のTabularPredictorを初期化
predictor_working = TabularPredictor(
    label='作業時間',                    # 予測対象の列名
    eval_metric='mean_absolute_error',   # 評価指標：平均絶対誤差
    problem_type='regression'            # 回帰問題として設定
).fit(
    train_data[features + ['作業時間']],  # 特徴量 + 目的変数
    time_limit=7200,                     # 学習時間の上限（2時間）
    presets='best_quality'               # 最高品質のプリセットを使用
)

print("作業時間予測モデルの学習完了")

# =======================================
# 8. AutoGluonモデルの学習（付帯作業時間）
# =======================================
print("\n=== 付帯作業時間予測モデルの学習開始 ===")

# 付帯作業時間予測用のTabularPredictorを初期化
predictor_ancillary = TabularPredictor(
    label='付帯作業時間',                # 予測対象の列名
    eval_metric='mean_absolute_error',   # 評価指標：平均絶対誤差
    problem_type='regression'            # 回帰問題として設定
).fit(
    train_data[features + ['付帯作業時間']],  # 特徴量 + 目的変数
    time_limit=7200,                      # 学習時間の上限（2時間）
    presets='best_quality'                # 最高品質のプリセットを使用
)

print("付帯作業時間予測モデルの学習完了")

# =======================================
# 9. テストデータでの予測
# =======================================
print("\n=== テストデータでの予測実行 ===")

# 作業時間の予測
preds_working = predictor_working.predict(test_data[features])
print(f"作業時間予測完了: {len(preds_working)}件")

# 付帯作業時間の予測
preds_ancillary = predictor_ancillary.predict(test_data[features])
print(f"付帯作業時間予測完了: {len(preds_ancillary)}件")

# =======================================
# 10. 提出ファイルの作成
# =======================================
print("\n=== 提出ファイルの作成 ===")

# 予測結果をDataFrameに整理
submission = pd.DataFrame({
    'index': test_data['index'],
    '作業時間予測': preds_working,
    '付帯作業時間予測': preds_ancillary
})

# CSVファイルとして保存（ヘッダーとインデックスなし）
submission.to_csv('submission.csv', index=False, header=False)

print("提出ファイル'submission.csv'が作成されました")
print(f"予測結果サンプル:")
print(submission.head())

# 予測結果の統計情報を表示
print(f"\n=== 予測結果の統計情報 ===")
print(f"作業時間予測の統計:")
print(f"  平均: {preds_working.mean():.2f}")
print(f"  最小: {preds_working.min():.2f}")
print(f"  最大: {preds_working.max():.2f}")

print(f"\n付帯作業時間予測の統計:")
print(f"  平均: {preds_ancillary.mean():.2f}")
print(f"  最小: {preds_ancillary.min():.2f}")
print(f"  最大: {preds_ancillary.max():.2f}")

# =======================================
# 11. モデル分析用コード（オプション）
# =======================================
# 以下のコードはコメントアウトされていますが、モデル分析に使用可能です

# # 作業時間モデルの学習概要を表示
# working_summary = predictor_working.fit_summary()
# print("=== 作業時間モデル学習概要 ===")
# print(working_summary)

# # 付帯作業時間モデルの学習概要を表示
# ancillary_summary = predictor_ancillary.fit_summary()
# print("=== 付帯作業時間モデル学習概要 ===")
# print(ancillary_summary)

# # 特徴量重要度の計算と表示
# working_feature_importance = predictor_working.feature_importance(data=train_data)
# ancillary_feature_importance = predictor_ancillary.feature_importance(data=train_data)

# print("=== 作業時間予測の重要特徴量トップ10 ===")
# print(working_feature_importance.head(10))

# print("=== 付帯作業時間予測の重要特徴量トップ10 ===")
# print(ancillary_feature_importance.head(10))

# # 特徴量重要度の可視化
# import matplotlib.pyplot as plt
# working_feature_importance.head(10).plot(kind='barh', figsize=(10, 6))
# plt.title('作業時間予測：特徴量重要度')
# plt.show()

# # モデルのリーダーボード表示
# working_leaderboard = predictor_working.leaderboard()
# ancillary_leaderboard = predictor_ancillary.leaderboard()

# print("=== 作業時間予測モデルリーダーボード ===")
# print(working_leaderboard)

# print("=== 付帯作業時間予測モデルリーダーボード ===")
# print(ancillary_leaderboard)

# # バリデーションデータでの性能評価
# validation_performance_working = predictor_working.evaluate(train_data[features + ['作業時間']])
# validation_performance_ancillary = predictor_ancillary.evaluate(train_data[features + ['付帯作業時間']])

# print("=== 作業時間予測モデル バリデーション性能 ===")
# print(validation_performance_working)

# print("=== 付帯作業時間予測モデル バリデーション性能 ===")
# print(validation_performance_ancillary)

print("\n=== 処理完了 ===")
