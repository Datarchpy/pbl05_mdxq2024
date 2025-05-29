# PBL05データクレンジング処理
# データフィルタ処理したデータをもとにデータ加工を行う

# ===== 処理内容の詳細説明 =====
"""
【データクレンジング処理の詳細】

1. グルアーデータ（糊付け機データ）の処理:
   - 28列 → 28列（列の内容を整理・修正）
   - 主な処理:
     * 列名の日本語化（カテゴリ名2 → 用紙種別）
     * 流用受注の有無をフラグ化
     * 新フォーマット（2020年2月4日以降）の付帯作業時間を修正
     * 日付・時刻データの統一的なパース処理
     * NULL値の適切な処理

2. 印刷機データの処理:
   - 42列 → 38列（不要列を削除し、内容を整理）
   - 主な処理:
     * 列名の日本語化（カテゴリ名1 → 粘着タイプ、カテゴリ名2 → 用紙種別）
     * 数量列の整理（数量1-3 → 通し数、色数(一般)、色数(特殊)）
     * 入力ミスと思われる数量4列の削除
     * 流用受注の有無をフラグ化
     * 新フォーマットの付帯作業時間を修正
     * 作業実績なし（所要時間=0）のデータを除外
     * 日付・時刻データの統一的なパース処理

3. データ品質向上のための処理:
   - 「« NULL »」文字列をNaN値に統一
   - 複数の日付フォーマットに対応したパース処理
   - 作業時間の整合性チェック
   - 異常値の特定と処理

4. 新フォーマット対応:
   - 2020年2月4日以降のデータで付帯作業時間の計算方法が変更
   - 旧フォーマット: 付帯作業時間 = 実際の付帯作業時間
   - 新フォーマット: 付帯作業時間 = 作業時間 + 残業時間（誤記録）
   - 正しい付帯作業時間 = 所要時間 - (作業時間 + 残業時間)

【注意点】
- Ydata-Profilingレポートは時間がかかるため、必要に応じて実行してください
"""


# ===== 初期設定 =====

# 日本語matplotlib表示のためのライブラリをインストール
! pip install japanize_matplotlib

import importlib
import sys
import subprocess

# Google Colab 上で実行しているかどうかを判断するフラグ
ON_COLAB = "google.colab" in sys.modules
print(f"ON_COLAB: {ON_COLAB}")

# Google Colab環境での設定
if ON_COLAB:
    # Google Drive にマウント
    drive = importlib.import_module("google.colab.drive")
    drive.mount("/content/drive/")
    
    # 必要なライブラリをインストール
    result = subprocess.run(
        ["pip", "install", "pygwalker", "ydata-profiling", "japanize_matplotlib"],
        capture_output=True,
        text=True,
    )
    print(result.stdout)

# パス設定（環境に応じて切り替え）
import os

if ON_COLAB:
    # Google Colabを利用する場合
    path = '/content/drive/MyDrive/DXQuest/'  # 適宜変更してください
else:
    # ローカルの場合
    path = "./"  # 適宜変更してください

# 必要ライブラリのインポート
import pandas as pd
import numpy as np
import japanize_matplotlib

# DataFrame表示で列を省略しないように設定
pd.set_option('display.max_columns', None)

# ===== データ出力フォルダの作成 =====

def makedir_helper(base_path, dir_name):
    """
    指定されたベースパスにディレクトリを作成するヘルパー関数
    
    Args:
        base_path: ベースパス
        dir_name: 作成するディレクトリ名
    
    Returns:
        作成されたディレクトリのフルパス
    """
    make_dir_path = os.path.join(base_path, dir_name)
    os.makedirs(make_dir_path, exist_ok=True)
    return make_dir_path

# データクレンジング結果を保存するディレクトリを作成
dc_data_dir = makedir_helper(path, 'data_cleaning')

# ===== データ読み込み =====

# フィルタ済みデータの読み込み元パス設定
read_base_path = os.path.join('/kaggle/input/pbl05-eda-data/', "eda_data/div_data")

# フィルタ済みデータの読み込み
# g: グルアー（糊付け機）データ、p: 印刷機データ
train_g = pd.read_csv(os.path.join(read_base_path, 'train_g_f.csv'))  # 訓練用グルアーデータ
train_p = pd.read_csv(os.path.join(read_base_path, 'train_p_f.csv'))  # 訓練用印刷機データ
test_g = pd.read_csv(os.path.join(read_base_path, 'test_g_f.csv'))    # テスト用グルアーデータ
test_p = pd.read_csv(os.path.join(read_base_path, 'test_p_f.csv'))    # テスト用印刷機データ

# 元データも読み込み（参照用）
base_train = pd.read_csv(os.path.join('/kaggle/input/pbl05-data/3-2_PBL05_data', 'train/base_train.csv'))
processing_train = pd.read_csv(os.path.join('/kaggle/input/pbl05-data/3-2_PBL05_data', 'train/processing_train.csv'))
actual_train = pd.read_csv(os.path.join('/kaggle/input/pbl05-data/3-2_PBL05_data', 'train/actual_train.csv'))

# ===== グルアーデータの処理 =====

print("=== グルアーデータの列構成確認 ===")
print(f"訓練用グルアーデータの列数: {len(train_g.columns)}")
print(f"テスト用グルアーデータの列数: {len(test_g.columns)}")

# 列名の見直し
# "Unnamed: 0" を "index" に変更
# "カテゴリ名2" を "用紙種別" に変更（より直感的な名前に）
train_g_1 = train_g.rename(columns={"Unnamed: 0": "index", "カテゴリ名2": "用紙種別"})
test_g_0 = test_g.drop(columns=["Unnamed: 0"])  # テストデータから不要なindex列を削除
test_g_1 = test_g_0.rename(columns={"カテゴリ名2": "用紙種別"})

# === 流用受注データの処理 ===
# Stitchさんからのヒント：流用受注〜 に注目

print("=== 流用受注データの分析 ===")
print("訓練用グルアーデータの流用受注情報:")
train_g_1[["流用受注日", "流用受注番号", "流用受注番号_upper"]].info()
print("テスト用グルアーデータの流用受注情報:")
test_g_1[["流用受注日", "流用受注番号", "流用受注番号_upper"]].info()

# 流用受注の有無フラグを作成
train_g_1_ex = train_g_1.copy()
test_g_1_ex = test_g_1.copy()

# 流用受注日がNaNでない場合は1、NaNの場合は0のフラグを作成
train_g_1_ex["流用受注有無"] = train_g_1["流用受注日"].apply(lambda x: 0 if pd.isna(x) else 1)
test_g_1_ex["流用受注有無"] = test_g_1["流用受注日"].apply(lambda x: 0 if pd.isna(x) else 1)

print("訓練用データの流用受注有無分布:")
print(train_g_1_ex["流用受注有無"].value_counts())
print("テスト用データの流用受注有無分布:")
print(test_g_1_ex["流用受注有無"].value_counts())

# === 不要な列の削除 ===
# base_XXX にあるデータなので使わない
remove_base_cols = ["仕上寸法区分", "受注数量", "実績数量", "納期", "頁数", "流用受注日", "流用受注番号", "流用受注番号_upper"]

# 予想に使えると思えない
remove_pred_cols = ["受注日", "製品仕様コード", "製品仕様コード_base"]

train_g_2 = train_g_1_ex.drop(columns=remove_base_cols)
train_g_3 = train_g_2.drop(columns=remove_pred_cols)

test_g_2 = test_g_1_ex.drop(columns=remove_base_cols)
test_g_3 = test_g_2.drop(columns=remove_pred_cols)

# === 数量1、数量項目名1の見直し ===
# 数量項目名1の内容を確認
print("数量項目名1の値分布:")
print(train_g["数量項目名1"].value_counts())

# 「枚数」が圧倒的多数なので、数量1を「枚数」に列名変更
train_g_4 = train_g_3.drop(columns=["数量項目名1"])
train_g_5 = train_g_4.rename(columns={"数量1": "枚数"})

test_g_4 = test_g_3.drop(columns=["数量項目名1"])
test_g_5 = test_g_4.rename(columns={"数量1": "枚数"})

# === 新フォーマット対応処理 ===
# 2020年2月4日以降の新フォーマットデータの処理

# 作業日_datetimeは不要なので削除
train_g_6 = train_g_5.drop(columns=["作業日_datetime"])
test_g_6 = test_g_5.copy()

# 日付列をdatetime型に変換
train_g_7 = train_g_6.copy()
train_g_7["作業日"] = pd.to_datetime(train_g_7["作業日"])
train_g_7["作成日"] = pd.to_datetime(train_g_7["作成日"])
test_g_7 = test_g_6.copy()
test_g_7["作業日"] = pd.to_datetime(test_g_7["作業日"])

# 作業時間の計算列を追加
train_g_8 = train_g_7.copy()
train_g_8["作業時間 + 残業時間"] = train_g_8["作業時間"] + train_g_8["残業時間"]
train_g_8["fix付帯作業時間"] = train_g_8["付帯作業時間"]  # 修正用の付帯作業時間列
test_g_8 = test_g_7.copy()

# === 新フォーマットの付帯作業時間修正 ===
# 新フォーマット且つ「作業時間 + 残業時間」と「付帯作業時間」が一致する行を特定
matching_g_rows = train_g_8[
    (train_g_8["新フォーマット"] == 1)
    & (train_g_8["作業時間 + 残業時間"] == train_g_8["付帯作業時間"])
]

print(f"新フォーマットで付帯作業時間の修正が必要な行数: {len(matching_g_rows)}")

# 該当行の付帯作業時間を「計算_所要時間-(作業時間+残業時間)」の値に修正
train_g_8.loc[
    (train_g_8["新フォーマット"] == 1)
    & (train_g_8["作業時間 + 残業時間"] == train_g_8["付帯作業時間"]),
    "fix付帯作業時間",
] = train_g_8.loc[
    (train_g_8["新フォーマット"] == 1)
    & (train_g_8["作業時間 + 残業時間"] == train_g_8["付帯作業時間"]),
    "計算_所要時間-(作業時間+残業時間)",
]

# 修正後の差分確認
different_g_values = train_g_8[train_g_8["fix付帯作業時間"] != train_g_8["付帯作業時間"]]
print(f"修正された行数: {len(different_g_values)}")

# データクレンジング済みグルアーデータとして設定
train_g_dc = train_g_8
test_g_dc = test_g_8

# === データ品質チェック ===

def view_non_numeric_rows(df, column_name):
    """
    指定列で数値でない行を取得する関数
    
    Args:
        df: DataFrame
        column_name: 列名
    
    Returns:
        数値でない行のSeries
    """
    non_numeric_rows = df[pd.to_numeric(df[column_name], errors='coerce').isnull()]
    return non_numeric_rows[column_name]

def replace_value_with_nan(df, column_name, data):
    """
    DataFrameの指定された列のデータが指定された値と一致する場合に、np.NaNに置き換える
    
    Args:
        df: DataFrame
        column_name: 列名
        data: 置き換え対象の値
    
    Returns:
        変更後のDataFrame
    """
    df.loc[df[column_name] == data, column_name] = np.NaN
    return df

# 数値列の「« NULL »」をNaNに置き換え
print("=== 数値列のNULL値処理 ===")
for col in ["展開寸法幅", "展開寸法長さ", "枚数"]:
    null_count = view_non_numeric_rows(train_g_dc, col).value_counts()
    if len(null_count) > 0:
        print(f"{col}のNULL値: {null_count}")
        train_g_dc = replace_value_with_nan(train_g_dc, col, "« NULL »")

# === 日付・時刻列の処理 ===

def parse_dates_with_formats(date_str):
    """
    複数の日付フォーマットに対応した日付パース関数
    
    Args:
        date_str: 日付文字列
    
    Returns:
        パース済みのdatetimeオブジェクト、またはNone
    """
    for fmt in ("%Y-%m-%d %H:%M:%S", "%m/%d/%Y %H:%M"):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except ValueError:
            continue
    return None  # どのフォーマットにもマッチしない場合

# 時刻列の日付パース
train_g_dc["開始時刻"] = train_g_dc["開始時刻"].apply(parse_dates_with_formats)
train_g_dc["終了時刻"] = train_g_dc["終了時刻"].apply(parse_dates_with_formats)
train_g_dc["下版予定日"] = train_g_dc["下版予定日"].apply(parse_dates_with_formats)

# 作業時間の計算
train_g_dc["終了時刻 - 開始時刻"] = train_g_dc["終了時刻"] - train_g_dc["開始時刻"]
train_g_dc["終了時刻 - 開始時刻(分)"] = train_g_dc["終了時刻 - 開始時刻"] / pd.to_timedelta("1m")
train_g_dc["diff_所要時間"] = train_g_dc["所要時間"] - train_g_dc["終了時刻 - 開始時刻(分)"]

# 時間差異のあるデータをチェック
diff_time_g = train_g_dc[train_g_dc["diff_所要時間"] != 0]
print(f"所要時間と実際の時間に差異のある行数: {len(diff_time_g)}")

print(f"グルアーデータ処理完了 - 最終列数: {len(train_g_dc.columns)}")

# ===== 印刷機データの処理 =====

print("\n=== 印刷機データの列構成確認 ===")
print(f"訓練用印刷機データの列数: {len(train_p.columns)}")
print(f"テスト用印刷機データの列数: {len(test_p.columns)}")

# 列名の見直し
# "Unnamed: 0" を "index" に変更
# "カテゴリ名1" を "粘着タイプ" に変更
# "カテゴリ名2" を "用紙種別" に変更
train_p_1 = train_p.rename(columns={"Unnamed: 0": "index", "カテゴリ名1": "粘着タイプ", "カテゴリ名2": "用紙種別"})
test_p_0 = test_p.drop(columns=["Unnamed: 0"])
test_p_1 = test_p_0.rename(columns={"カテゴリ名1": "粘着タイプ", "カテゴリ名2": "用紙種別"})

# === 流用受注データの処理（印刷機版） ===
print("=== 印刷機データの流用受注情報 ===")
print("訓練用印刷機データの流用受注情報:")
train_p_1[["流用受注日", "流用受注番号", "流用受注番号_upper"]].info()
print("テスト用印刷機データの流用受注情報:")
test_p_1[["流用受注日", "流用受注番号", "流用受注番号_upper"]].info()

# 流用受注の有無フラグを作成
train_p_1_ex = train_p_1.copy()
test_p_1_ex = test_p_1.copy()

train_p_1_ex["流用受注有無"] = train_p_1["流用受注日"].apply(lambda x: 0 if pd.isna(x) else 1)
test_p_1_ex["流用受注有無"] = test_p_1["流用受注日"].apply(lambda x: 0 if pd.isna(x) else 1)

# === 不要な列の削除（印刷機版） ===
# base_XXX にあるデータなので使わない
remove_base_cols_p = ["流用受注日", "流用受注番号", "流用受注番号_upper"]

# 予想に使えると思えない
remove_pred_cols_p = ["受注日", "受注番号", "受注番号_upper", "製品仕様コード", "製品仕様コード_base"]

train_p_2 = train_p_1_ex.drop(columns=remove_base_cols_p)
train_p_3 = train_p_2.drop(columns=remove_pred_cols_p)

test_p_2 = test_p_1_ex.drop(columns=remove_base_cols_p)
test_p_3 = test_p_2.drop(columns=remove_pred_cols_p)

# === 数量4の確認と削除 ===
print("=== 数量4の内容確認 ===")
print("訓練用データの数量4分布:")
print(train_p["数量4"].value_counts())
print(f"訓練用データの数量4のNaN数: {train_p['数量4'].isna().sum()}")

print("テスト用データの数量4分布:")
print(test_p["数量4"].value_counts())
print(f"テスト用データの数量4のNaN数: {test_p['数量4'].isna().sum()}")

# 数量4は入力ミスと判断し削除
train_p_4 = train_p_3.drop(columns=["数量4"])
test_p_4 = test_p_3.drop(columns=["数量4"])

# === 数量1、数量2、数量3の見直し ===
# フィルタ処理で判明した項目名に基づいて列名を変更
# 数量項目名1: ['通し数']
# 数量項目名2: ['色数(一般)']
# 数量項目名3: ['色数(特殊)']
train_p_5 = train_p_4.rename(columns={"数量1": "通し数", "数量2": "色数(一般)", "数量3": "色数(特殊)"})
test_p_5 = test_p_4.rename(columns={"数量1": "通し数", "数量2": "色数(一般)", "数量3": "色数(特殊)"})

# === 新フォーマット対応処理（印刷機版） ===
# 作業日_datetimeは不要なので削除
train_p_6 = train_p_5.drop(columns=["作業日_datetime"])
test_p_6 = test_p_5.copy()

# 日付列をdatetime型に変換
train_p_7 = train_p_6.copy()
train_p_7["作業日"] = pd.to_datetime(train_p_7["作業日"])
train_p_7["作成日"] = pd.to_datetime(train_p_7["作成日"])
test_p_7 = test_p_6.copy()
test_p_7["作業日"] = pd.to_datetime(test_p_7["作業日"])

# 作業時間の計算列を追加
train_p_8 = train_p_7.copy()
train_p_8["作業時間 + 残業時間"] = train_p_8["作業時間"] + train_p_8["残業時間"]
train_p_8["fix付帯作業時間"] = train_p_8["付帯作業時間"]  # 修正用の付帯作業時間列
test_p_8 = test_p_7.copy()

# === 新フォーマットの付帯作業時間修正（印刷機版） ===
matching_p_rows = train_p_8[
    (train_p_8["新フォーマット"] == 1)
    & (train_p_8["作業時間 + 残業時間"] == train_p_8["付帯作業時間"])
]

print(f"印刷機データで付帯作業時間の修正が必要な行数: {len(matching_p_rows)}")

# 該当行の付帯作業時間を修正
train_p_8.loc[
    (train_p_8["新フォーマット"] == 1)
    & (train_p_8["作業時間 + 残業時間"] == train_p_8["付帯作業時間"]),
    "fix付帯作業時間",
] = train_p_8.loc[
    (train_p_8["新フォーマット"] == 1)
    & (train_p_8["作業時間 + 残業時間"] == train_p_8["付帯作業時間"]),
    "計算_所要時間-(作業時間+残業時間)",
]

# 修正後の差分確認
different_p_values = train_p_8[train_p_8["fix付帯作業時間"] != train_p_8["付帯作業時間"]]
print(f"印刷機データで修正された行数: {len(different_p_values)}")

# データクレンジング済み印刷機データとして設定
train_p_dc = train_p_8
test_p_dc = test_p_8

# === データ品質チェック（印刷機版） ===
print("=== 印刷機データの数値列NULL値処理 ===")

# 数値列の「« NULL »」をNaNに置き換え
for col in ["通し数", "色数(一般)", "色数(特殊)"]:
    null_count = view_non_numeric_rows(train_p_dc, col).value_counts()
    if len(null_count) > 0:
        print(f"{col}のNULL値: {null_count}")
        train_p_dc = replace_value_with_nan(train_p_dc, col, "« NULL »")

# === 日付・時刻列の処理（印刷機版） ===
# 時刻列の日付パース
train_p_dc["開始時刻"] = train_p_dc["開始時刻"].apply(parse_dates_with_formats)
train_p_dc["終了時刻"] = train_p_dc["終了時刻"].apply(parse_dates_with_formats)

# 作業時間の計算
train_p_dc["終了時刻 - 開始時刻"] = train_p_dc["終了時刻"] - train_p_dc["開始時刻"]
train_p_dc["終了時刻 - 開始時刻(分)"] = train_p_dc["終了時刻 - 開始時刻"] / pd.to_timedelta("1m")
train_p_dc["diff_所要時間"] = train_p_dc["所要時間"] - train_p_dc["終了時刻 - 開始時刻(分)"]

# 時間差異のあるデータをチェック
diff_time_p = train_p_dc[train_p_dc["diff_所要時間"] != 0]
print(f"印刷機データで所要時間と実際の時間に差異のある行数: {len(diff_time_p)}")

# 所要時間が0の行を除外（作業実績なしと判断）
print(f"処理前の印刷機データ行数: {train_p_dc.shape}")
train_p_dc = train_p_dc[train_p_dc["所要時間"] != 0]
print(f"処理後の印刷機データ行数: {train_p_dc.shape}")

print(f"印刷機データ処理完了 - 最終列数: {len(train_p_dc.columns)}")

# ===== 最終データの出力 =====

print("\n=== 最終データ情報 ===")
print(f"訓練用グルアーデータ: {train_g_dc.shape[0]}行 × {train_g_dc.shape[1]}列")
print(f"訓練用印刷機データ: {train_p_dc.shape[0]}行 × {train_p_dc.shape[1]}列")
print(f"テスト用グルアーデータ: {test_g_dc.shape[0]}行 × {test_g_dc.shape[1]}列")
print(f"テスト用印刷機データ: {test_p_dc.shape[0]}行 × {test_p_dc.shape[1]}列")

# CSVファイルとして出力
train_g_dc.to_csv(os.path.join(dc_data_dir, "train_g_dc.csv"), index=False)
train_p_dc.to_csv(os.path.join(dc_data_dir, "train_p_dc.csv"), index=False)
test_g_dc.to_csv(os.path.join(dc_data_dir, "test_g_dc.csv"), index=False)
test_p_dc.to_csv(os.path.join(dc_data_dir, "test_p_dc.csv"), index=False)

print("データクレンジング処理が完了しました。")
print(f"出力先: {dc_data_dir}")

# ===== 出力データの再読み込み確認 =====
# CSVとして出力したデータを再読み込みして確認
train_g_dc_csv = pd.read_csv(os.path.join(dc_data_dir, "train_g_dc.csv"))
train_p_dc_csv = pd.read_csv(os.path.join(dc_data_dir, "train_p_dc.csv"))
test_g_dc_csv = pd.read_csv(os.path.join(dc_data_dir, "test_g_dc.csv"))
test_p_dc_csv = pd.read_csv(os.path.join(dc_data_dir, "test_p_dc.csv"))

print("\n=== CSV再読み込み確認 ===")
print("再読み込み完了")

# ===== Ydata-Profiling レポート生成（オプション） =====
# データを眺める用のプロファイリングレポート生成
# 注意: 実行には時間がかかるため、必要に応じてコメントアウトを解除

"""
from ydata_profiling import ProfileReport
import matplotlib
matplotlib.rc('font', family='IPAPGothic')  # 全体のフォントをIPAゴシックに設定

# プロファイルレポートの生成
profile_train_g = ProfileReport(train_g_dc_csv, title="train_g_dc.csv Profiling Report")
profile_train_p = ProfileReport(train_p_dc_csv, title="train_p_dc.csv Profiling Report")
profile_test_g = ProfileReport(test_g_dc_csv, title="test_g_dc.csv Profiling Report")
profile_test_p = ProfileReport(test_p_dc_csv, title="test_p_dc.csv Profiling Report")

# レポート出力設定
view_report_notebook = False  # レポートをノートブックで表示する場合はTrue
report_to_file = True  # レポートをhtml出力する場合はTrue

def ydata_report(profile, file_path):
    \"\"\"Ydata-Profilingレポートの出力\"\"\"
    if view_report_notebook:
        profile.to_notebook_iframe()
    if report_to_file:
        profile.to_file(file_path)

# HTMLレポートファイルの出力
# ydata_report(profile_train_g, os.path.join(dc_data_dir, "train_g_dc.html"))
# ydata_report(profile_train_p, os.path.join(dc_data_dir, "train_p_dc.html"))
# ydata_report(profile_test_g, os.path.join(dc_data_dir, "test_g_dc.html"))
# ydata_report(profile_test_p, os.path.join(dc_data_dir, "test_p_dc.html"))
"""

print("全ての処理が完了しました！")
print("\n=== 処理サマリー ===")
print("1. 環境設定とライブラリインポート")
print("2. フィルタ済みデータの読み込み")
print("3. グルアーデータのクレンジング処理")
print("   - 列名の統一")
print("   - 流用受注フラグの作成")
print("   - 不要列の削除")
print("   - 新フォーマット対応")
print("   - 付帯作業時間の修正")
print("   - 日付・時刻データの処理")
print("4. 印刷機データのクレンジング処理")
print("   - 列名の統一")
print("   - 流用受注フラグの作成")
print("   - 不要列の削除")
print("   - 数量列の整理")
print("   - 新フォーマット対応")
print("   - 付帯作業時間の修正")
print("   - 日付・時刻データの処理")
print("   - 無効データ（所要時間=0）の除外")
print("5. クレンジング済みデータのCSV出力")
print("6. データ品質確認")
print("\n出力ファイル:")
print("- train_g_dc.csv: 訓練用グルアーデータ")
print("- train_p_dc.csv: 訓練用印刷機データ")
print("- test_g_dc.csv: テスト用グルアーデータ")
print("- test_p_dc.csv: テスト用印刷機データ")
