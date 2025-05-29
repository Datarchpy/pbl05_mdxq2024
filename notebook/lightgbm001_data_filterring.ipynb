# PBL05 データフィルタリング・前処理ノートブック

# ====================================
# 初期設定
# ====================================

# Google Colabで日本語フォント表示用ライブラリをインストール
! pip install japanize_matplotlib

import importlib
import sys
import subprocess

# Google Colab上で実行しているかどうかを判断するフラグ
ON_COLAB = "google.colab" in sys.modules
print(f"ON_COLAB: {ON_COLAB}")

if ON_COLAB:
    # Google Drive にマウントする
    drive = importlib.import_module("google.colab.drive")
    drive.mount("/content/drive/")

    # 必要なライブラリをインストール
    # Notebook以外で実行することも考慮し、Pythonコードで実行
    result = subprocess.run(
        ["pip", "install", "pygwalker", "ydata-profiling", "japanize_matplotlib"], # 適宜変更してください
        capture_output=True,
        text=True,
    )
    print(result.stdout)

import os

# 実行環境に応じてパスを設定
if ON_COLAB:
    # Google Colabを利用する場合
    path = '/content/drive/MyDrive/DXQuest/' # 適宜変更してください
else:
    # ローカルの場合
    path = "./" # 適宜変更してください

# 必要なライブラリをインポート
import pandas as pd
import japanize_matplotlib

# ====================================
# ユーティリティ関数定義
# ====================================

def makedir_helper(base_path: path, dir_name: str):
    """ディレクトリを作成するヘルパー関数"""
    make_dir_path = os.path.join(base_path, dir_name)
    os.makedirs(make_dir_path, exist_ok=True)
    return make_dir_path

def file_exist_check(base_path, offset_path):
    """ファイルの存在チェック関数"""
    file_path = os.path.join(base_path, offset_path)
    if os.path.exists(file_path):
        if os.path.isfile(file_path):
            return
    raise ValueError(f"{offset_path} が見つかりません")

def check_delete_columns(df1, df2, df1_name, df2_name):
    """2つのDataFrameの列数変化を確認する関数"""
    print(f"{df1_name: >20}.shape : {df1.shape}")
    print(f"{df2_name: >20}.shape : {df2.shape}")
    set_df_1 = set(df1.columns)
    set_df_2 = set(df2.columns)
    print(set_df_1 - set_df_2)

def remove_constant_columns(df, df_name):
    """
    DataFrame型引数dfの列の内、すべての行のデータが同じものがあれば削除して戻り値でDataFrameを返す
    """
    print(df_name)
    print("=" * 40)

    for col in df.columns:
        col_unique = df[col].unique()
        # すべての値が同じ（ユニーク値が1つ）の列を削除
        if len(col_unique) == 1:
            print(f"{col} : {col_unique}")
            df = df.drop(col, axis=1)

    print("-" * 40)
    return df

# ====================================
# 成果物出力用ディレクトリ作成
# ====================================

# 各種データ出力用ディレクトリを作成
eda_data_dir = makedir_helper(path, 'eda_data')

# ====================================
# ファイル存在確認
# ====================================

# 必要なCSVファイルが存在するかチェック
file_exist_check('/kaggle/input/pbl05-data/3-2_PBL05_data', 'train/actual_train.csv')
file_exist_check('/kaggle/input/pbl05-data/3-2_PBL05_data', 'train/base_train.csv')
file_exist_check('/kaggle/input/pbl05-data/3-2_PBL05_data', 'train/processing_train.csv')
file_exist_check('/kaggle/input/pbl05-data/3-2_PBL05_data', 'test/actual_test.csv')
file_exist_check('/kaggle/input/pbl05-data/3-2_PBL05_data', 'test/base_test.csv')
file_exist_check('/kaggle/input/pbl05-data/3-2_PBL05_data', 'test/processing_test.csv')

# DataFrame表示で列を省略しないように設定
pd.set_option('display.max_columns', None)

# ====================================
# データ読み込み
# ====================================

# 各CSVファイルを読み込み
work_data01_df = pd.read_csv(os.path.join('/kaggle/input/pbl05-data/3-2_PBL05_data', 'train/actual_train.csv'))
work_data02_df = pd.read_csv(os.path.join('/kaggle/input/pbl05-data/3-2_PBL05_data', 'train/base_train.csv'))
work_data03_df = pd.read_csv(os.path.join('/kaggle/input/pbl05-data/3-2_PBL05_data', 'train/processing_train.csv'))

work_data11_df = pd.read_csv(os.path.join('/kaggle/input/pbl05-data/3-2_PBL05_data', 'test/actual_test.csv'))
work_data12_df = pd.read_csv(os.path.join('/kaggle/input/pbl05-data/3-2_PBL05_data', 'test/base_test.csv'))
work_data13_df = pd.read_csv(os.path.join('/kaggle/input/pbl05-data/3-2_PBL05_data', 'test/processing_test.csv'))

# ====================================
# 列構造の分析
# ====================================

# actual_train.csv と actual_test.csv の列数が異なることを確認
assert len(work_data01_df.columns) != len(work_data11_df.columns), "actual_train.csv と actual_test.csv は列数が違う"

# 各ファイルの列情報を辞書に格納
actual_train_columns = {'actual_train.csv': work_data01_df.columns}
actual_test_columns = {'actual_test.csv': work_data11_df.columns}

# base系ファイルの列一致確認
base_columns = {
    'base_train.csv': work_data02_df.columns,
    'base_test.csv': work_data12_df.columns,
}
assert len(work_data02_df.columns) == len(work_data12_df.columns), "列数一致"

# 列の差異をチェック（base系）
for col in work_data02_df.columns:
    if col not in work_data12_df.columns:
        print(col)

for col in work_data12_df.columns:
    if col not in work_data02_df.columns:
        print(col)

# processing系ファイルの列一致確認
processing_columns = {
    'processing_train.csv': work_data03_df.columns,
    'processing_test.csv': work_data13_df.columns,
}
assert len(work_data03_df.columns) == len(work_data13_df.columns), "列数一致"

# 列の差異をチェック（processing系）
for col in work_data03_df.columns:
    if col not in work_data13_df.columns:
        print(col)

for col in work_data13_df.columns:
    if col not in work_data03_df.columns:
        print(col)

# ====================================
# 列一覧の出力
# ====================================

# 列情報をDataFrameに変換
base_columns_df = pd.DataFrame(base_columns)
processing_columns_df = pd.DataFrame(processing_columns)
actual_train_columns_df = pd.DataFrame(actual_train_columns)
actual_test_columns_df = pd.DataFrame(actual_test_columns)

# 列データ用ディレクトリを作成
columns_data_dir = makedir_helper(eda_data_dir, 'columns_data')

# 列情報をCSVファイルとして出力
base_columns_df.to_csv(os.path.join(columns_data_dir, 'columns_base.csv'))
processing_columns_df.to_csv(os.path.join(columns_data_dir, 'columns_processing.csv'))
actual_train_columns_df.to_csv(os.path.join(columns_data_dir, 'columns_actual_train.csv'))
actual_test_columns_df.to_csv(os.path.join(columns_data_dir, 'columns_actual_test.csv'))

# ====================================
# データフィルタリング処理
# ====================================

# すべての値がNaNの列を削除
work_data01_df_w1 = work_data01_df.dropna(axis=1, how='all')
work_data02_df_w1 = work_data02_df.dropna(axis=1, how='all')
work_data03_df_w1 = work_data03_df.dropna(axis=1, how='all')

work_data11_df_w1 = work_data11_df.dropna(axis=1, how='all')
work_data12_df_w1 = work_data12_df.dropna(axis=1, how='all')
work_data13_df_w1 = work_data13_df.dropna(axis=1, how='all')

# 削除された列を確認
check_delete_columns(work_data01_df, work_data01_df_w1, 'work_data01_df', 'work_data01_df_w1')
check_delete_columns(work_data02_df, work_data02_df_w1, 'work_data02_df', 'work_data03_df_w1')
check_delete_columns(work_data03_df, work_data03_df_w1, 'work_data03_df', 'work_data02_df_w1')

check_delete_columns(work_data11_df, work_data11_df_w1, 'work_data11_df', 'work_data11_df_w1')
check_delete_columns(work_data12_df, work_data12_df_w1, 'work_data12_df', 'work_data12_df_w1')
check_delete_columns(work_data13_df, work_data13_df_w1, 'work_data13_df', 'work_data13_df_w1')

# 定数列（すべての値が同じ列）を削除
work_data01_df_w2 = remove_constant_columns(work_data01_df_w1, "work_data01_df_w1")
work_data02_df_w2 = remove_constant_columns(work_data02_df_w1, "work_data02_df_w1")
work_data03_df_w2 = remove_constant_columns(work_data03_df_w1, "work_data03_df_w1")

work_data11_df_w2 = remove_constant_columns(work_data11_df_w1, "work_data11_df_w1")
work_data12_df_w2 = remove_constant_columns(work_data12_df_w1, "work_data12_df_w1")
work_data13_df_w2 = remove_constant_columns(work_data13_df_w1, "work_data13_df_w1")

# 変更確認
check_delete_columns(work_data01_df_w1, work_data01_df_w2, 'work_data01_df_w1', 'work_data01_df_w2')
check_delete_columns(work_data02_df_w1, work_data02_df_w2, 'work_data02_df_w1', 'work_data03_df_w2')
check_delete_columns(work_data03_df_w1, work_data03_df_w2, 'work_data03_df_w1', 'work_data02_df_w2')

check_delete_columns(work_data11_df_w1, work_data11_df_w2, 'work_data11_df_w1', 'work_data11_df_w2')
check_delete_columns(work_data12_df_w1, work_data12_df_w2, 'work_data12_df_w1', 'work_data12_df_w2')
check_delete_columns(work_data13_df_w1, work_data13_df_w2, 'work_data13_df_w1', 'work_data13_df_w2')

# ====================================
# フィルタ済みデータの出力
# ====================================

# フィルタ済みデータ用ディレクトリを作成
filtered_data_dir = makedir_helper(eda_data_dir, 'filtered_data')

# フィルタ済みデータをCSVファイルとして出力
work_data01_df_w2.to_csv(os.path.join(filtered_data_dir, "actual_train_filtered.csv"))
work_data02_df_w2.to_csv(os.path.join(filtered_data_dir, "base_train_filtered.csv"))
work_data03_df_w2.to_csv(os.path.join(filtered_data_dir, "processing_train_filtered.csv"))

work_data11_df_w2.to_csv(os.path.join(filtered_data_dir, "actual_test_filtered.csv"))
work_data12_df_w2.to_csv(os.path.join(filtered_data_dir, "base_test_filtered.csv"))
work_data13_df_w2.to_csv(os.path.join(filtered_data_dir, "processing_test_filtered.csv"))

# ====================================
# フィルタ済みファイルで読み直し
# ====================================

# 保存したフィルタ済みファイルを再読み込み
df01 = pd.read_csv(os.path.join(filtered_data_dir, "actual_train_filtered.csv"), index_col=0)
df02 = pd.read_csv(os.path.join(filtered_data_dir, "base_train_filtered.csv"), index_col=0)
df03 = pd.read_csv(os.path.join(filtered_data_dir, "processing_train_filtered.csv"), index_col=0)

df11 = pd.read_csv(os.path.join(filtered_data_dir, "actual_test_filtered.csv"), index_col=0)
df12 = pd.read_csv(os.path.join(filtered_data_dir, "base_test_filtered.csv"), index_col=0)
df13 = pd.read_csv(os.path.join(filtered_data_dir, "processing_test_filtered.csv"), index_col=0)

# ====================================
# データ結合処理（受注番号そのまま）
# ====================================

# 結合データ用ディレクトリを作成
merge_data_dir = makedir_helper(eda_data_dir, 'merge_data')

def merge_proc_act(df_proc, df_act):
    """processing と actual データを結合する関数"""
    merge_df = pd.merge(df_proc, df_act, on=['受注番号','号機名'], suffixes=('_process', '_actual'))
    return merge_df

# processing と actual データを結合
merge_train_proc_act_df = merge_proc_act(df03, df01)
merge_test_proc_act_df = merge_proc_act(df13, df11)

# 結合済みデータを出力
merge_train_proc_act_df.to_csv(os.path.join(merge_data_dir, "merge_train_proc_act.csv"))
merge_test_proc_act_df.to_csv(os.path.join(merge_data_dir, "merge_test_proc_act.csv"))

# ====================================
# 予測対象機材のフィルタリング
# ====================================

def filer_machine_names(df, machine_names):
    """指定した機材名でデータをフィルタする関数"""
    return df[df['号機名'].isin(machine_names)]

# 予測対象の機材のみをフィルタ
merge_train_proc_act_df_filterd = filer_machine_names(merge_train_proc_act_df, ['グルアー','2号機','4号機','6号機','7号機','8号機'])
merge_test_proc_act_df_filterd  = filer_machine_names(merge_test_proc_act_df , ['グルアー','2号機','4号機','6号機','7号機','8号機'])

# フィルタ結果を確認
check_delete_columns(merge_train_proc_act_df, merge_train_proc_act_df_filterd, 'merge_train_proc_act_df', 'merge_train_proc_act_df_filterd')
check_delete_columns(merge_test_proc_act_df,  merge_test_proc_act_df_filterd,  'merge_test_proc_act_df',  'merge_test_proc_act_df_filterd')

# フィルタ後の定数列を削除
merge_train_proc_act_df_filterd_w = remove_constant_columns(merge_train_proc_act_df_filterd, "merge_train_proc_act_df_filterd")
merge_test_proc_act_df_filterd_w  = remove_constant_columns(merge_test_proc_act_df_filterd, "merge_test_proc_act_df_filterd")

# 変更確認
check_delete_columns(merge_train_proc_act_df_filterd, merge_train_proc_act_df_filterd_w, 'merge_train_proc_act_df_filterd', 'merge_train_proc_act_df_filterd_w')
check_delete_columns(merge_test_proc_act_df_filterd,  merge_test_proc_act_df_filterd_w,  'merge_test_proc_act_df_filterd',  'merge_test_proc_act_df_filterd_w')

# フィルタ済みデータを出力
merge_train_proc_act_df_filterd_w.to_csv(os.path.join(merge_data_dir, "merge_train_proc_act_filterd.csv"))
merge_test_proc_act_df_filterd_w.to_csv(os.path.join(merge_data_dir,  "merge_test_proc_act_filterd.csv"))

# ====================================
# base データとの結合
# ====================================

def merge_proc_act_base(df_proc_act, df_base):
    """processing+actual と base データを結合する関数"""
    merge_df = pd.merge(df_proc_act, df_base, how="left", on=['受注番号'], suffixes=('', '_base'))
    return merge_df

# base データと結合
merge_train = merge_proc_act_base(merge_train_proc_act_df_filterd_w, df02)
merge_test  = merge_proc_act_base(merge_test_proc_act_df_filterd_w,  df12)

# 結合済みデータを出力
merge_train.to_csv(os.path.join(merge_data_dir, "merge_train.csv"))
merge_test.to_csv(os.path.join(merge_data_dir,  "merge_test.csv"))

# 結合後の定数列を削除
merge_train_filterd = remove_constant_columns(merge_train, "merge_train")
merge_test_filterd  = remove_constant_columns(merge_test, "merge_test")

# 変更確認
check_delete_columns(merge_train, merge_train_filterd, 'merge_train', 'merge_train_filterd')
check_delete_columns(merge_test,  merge_test_filterd,  'merge_test',  'merge_test')

# 最終的な結合データを出力
merge_train_filterd.to_csv(os.path.join(merge_data_dir, "merge_train_filterd.csv"))
merge_test_filterd.to_csv(os.path.join(merge_data_dir,  "merge_test_filterd.csv"))

# 重複行の確認
print("重複行確認（train）:")
print(merge_train_filterd[merge_train_filterd.duplicated()])

print("重複行確認（test）:")
print(merge_test_filterd[merge_test_filterd.duplicated()])

# ====================================
# データ結合処理（受注番号大文字統一）
# ====================================

# 大文字統一版データ用ディレクトリを作成
merge_upper_data_dir = makedir_helper(eda_data_dir, 'merge_upper_data')

# 受注番号を大文字に統一
df01_upper = df01.copy()
df01_upper['受注番号_upper'] = df01_upper['受注番号'].str.upper()
df02_upper = df02.copy()
df02_upper['受注番号_upper'] = df02_upper['受注番号'].str.upper()
df02_upper['流用受注番号_upper'] = df02_upper['流用受注番号'].str.upper()
df03_upper = df03.copy()
df03_upper['受注番号_upper'] = df03_upper['受注番号'].str.upper()

df11_upper = df11.copy()
df11_upper['受注番号_upper'] = df11_upper['受注番号'].str.upper()
df12_upper = df12.copy()
df12_upper['受注番号_upper'] = df12_upper['受注番号'].str.upper()
df12_upper['流用受注番号_upper'] = df12_upper['流用受注番号'].str.upper()
df13_upper = df13.copy()
df13_upper['受注番号_upper'] = df13_upper['受注番号'].str.upper()

def merge_proc_act_upper(df_proc, df_act):
    """大文字統一した受注番号でprocessing と actual データを結合する関数"""
    merge_df = pd.merge(df_proc, df_act, on=['受注番号_upper','号機名'], suffixes=('_process', '_actual'))
    return merge_df

# 大文字統一版で結合
merge_upper_train_proc_act_df = merge_proc_act_upper(df03_upper, df01_upper)
merge_upper_test_proc_act_df = merge_proc_act_upper(df13_upper, df11_upper)

# 大文字統一版結合データを出力
merge_upper_train_proc_act_df.to_csv(os.path.join(merge_upper_data_dir, "merge_upper_train_proc_act.csv"))
merge_upper_test_proc_act_df.to_csv(os.path.join(merge_upper_data_dir, "merge_upper_test_proc_act.csv"))

# 予測対象機材でフィルタ
merge_upper_train_proc_act_df_filterd = filer_machine_names(merge_upper_train_proc_act_df, ['グルアー','2号機','4号機','6号機','7号機','8号機'])
merge_upper_test_proc_act_df_filterd  = filer_machine_names(merge_upper_test_proc_act_df,  ['グルアー','2号機','4号機','6号機','7号機','8号機'])

# フィルタ結果確認
check_delete_columns(merge_upper_train_proc_act_df, merge_upper_train_proc_act_df_filterd, 'merge_upper_train_proc_act_df', 'merge_upper_train_proc_act_df_filterd')
check_delete_columns(merge_upper_test_proc_act_df,  merge_upper_test_proc_act_df_filterd,  'merge_upper_test_proc_act_df',  'merge_upper_test_proc_act_df_filterd')

# 定数列削除
merge_upper_train_proc_act_df_filterd_w = remove_constant_columns(merge_upper_train_proc_act_df_filterd, "merge_upper_train_proc_act_df_filterd")
merge_upper_test_proc_act_df_filterd_w  = remove_constant_columns(merge_upper_test_proc_act_df_filterd, "merge_upper_test_proc_act_df_filterd")

# 変更確認
check_delete_columns(merge_upper_train_proc_act_df_filterd, merge_upper_train_proc_act_df_filterd_w, 'merge_upper_train_proc_act_df_filterd', 'merge_upper_train_proc_act_df_filterd_w')
check_delete_columns(merge_upper_test_proc_act_df_filterd,  merge_upper_test_proc_act_df_filterd_w,  'merge_upper_test_proc_act_df_filterd',  'merge_upper_test_proc_act_df_filterd_w')

# フィルタ済みデータを出力
merge_upper_train_proc_act_df_filterd_w.to_csv(os.path.join(merge_upper_data_dir, "merge_upper_train_proc_act_filterd.csv"))
merge_upper_test_proc_act_df_filterd_w.to_csv(os.path.join(merge_upper_data_dir,  "merge_upper_test_proc_act_filterd.csv"))

def merge_proc_act_base_upper(df_proc_act, df_base):
    """大文字統一した受注番号でprocessing+actual と base データを結合する関数"""
    merge_df = pd.merge(df_proc_act, df_base, how="left", on=['受注番号_upper'], suffixes=('', '_base'))
    return merge_df

# 大文字統一版でbase データと結合
merge_train_upper = merge_proc_act_base_upper(merge_upper_train_proc_act_df_filterd_w, df02_upper)
merge_test_upper  = merge_proc_act_base_upper(merge_upper_test_proc_act_df_filterd_w,  df12_upper)

# 大文字統一版結合データを出力
merge_train_upper.to_csv(os.path.join(merge_upper_data_dir, "merge_train_upper.csv"))
merge_test_upper.to_csv(os.path.join(merge_upper_data_dir,  "merge_test_upper.csv"))

# 定数列削除
merge_train_upper_filterd = remove_constant_columns(merge_train_upper, "merge_train_upper")
merge_test_upper_filterd  = remove_constant_columns(merge_test_upper, "merge_test_upper")

# 変更確認
check_delete_columns(merge_train_upper, merge_train_upper_filterd, 'merge_train_upper', 'merge_train_upper_filterd')
check_delete_columns(merge_test_upper,  merge_test_upper_filterd,  'merge_test_upper',  'merge_test_upper_filterd')

# 最終的な大文字統一版データを出力
merge_train_upper_filterd.to_csv(os.path.join(merge_upper_data_dir, "merge_train_upper_filterd.csv"))
merge_test_upper_filterd.to_csv(os.path.join(merge_upper_data_dir,  "merge_test_upper_filterd.csv"))

# 重複行の確認
print("重複行確認（train upper）:")
print(merge_train_upper_filterd[merge_train_upper_filterd.duplicated()])

print("重複行確認（test upper）:")
print(merge_test_upper_filterd[merge_test_upper_filterd.duplicated()])

# ====================================
# データ加工・分割処理
# ====================================

# 分割データ用ディレクトリを作成
div_data_dir = makedir_helper(eda_data_dir, 'div_data')

# 大文字統一版データを読み込み
merged_train_df = pd.read_csv(os.path.join(merge_upper_data_dir, "merge_train_upper_filterd.csv"), index_col=0)
merged_test_df  = pd.read_csv(os.path.join(merge_upper_data_dir, "merge_test_upper_filterd.csv"), index_col=0)

# ====================================
# 資料に従ったデータ加工（01_PBL05_exercise03.pdf ページ6 に従い加工）
# ====================================

# 作業日をdatetime型に変換
merged_train_df["作業日_datetime"] = pd.to_datetime(merged_train_df["作業日"])

# 新フォーマット列を追加（2020年2月4日以降を1、それ以前を0）
merged_train_df["新フォーマット"] = 0
merged_train_df.loc[merged_train_df["作業日_datetime"] >= pd.to_datetime("2020-02-04"), "新フォーマット"] = 1

# 所要時間から作業時間と残業時間の合計を引いた値を計算
merged_train_df["計算_所要時間-(作業時間+残業時間)"] = merged_train_df["所要時間"] - (merged_train_df["作業時間"] + merged_train_df["残業時間"])

# 付帯作業時間の計算（新旧フォーマットで異なる計算方法）
merged_train_df["付帯作業時間"] = merged_train_df["計算_所要時間-(作業時間+残業時間)"]  # 一旦コピー
# 新フォーマット(2020年2月4日以降)の行は合計時間で上書き
merged_train_df.loc[merged_train_df["新フォーマット"]==1, "付帯作業時間"] = merged_train_df["合計時間"]

# ====================================
# グルアー・印刷機でデータを分割
# ====================================

# グルアー（貼り機）と印刷機でデータを分割
train_g = merged_train_df[merged_train_df["号機名"]=="グルアー"]    # グルアー（貼り機）
train_p = merged_train_df[merged_train_df["号機名"]!="グルアー"]   # 印刷機

test_g = merged_test_df[merged_test_df["号機名"]=="グルアー"]      # グルアー（貼り機）
test_p = merged_test_df[merged_test_df["号機名"]!="グルアー"]     # 印刷機

# 各データセットで定数列を削除
train_g_filterd = remove_constant_columns(train_g, "train_g")
train_p_filterd = remove_constant_columns(train_p, "train_p")
test_g_filterd = remove_constant_columns(test_g, "test_g")
test_p_filterd = remove_constant_columns(test_p, "test_p")

# 分割済みデータを出力
train_g_filterd.to_csv(os.path.join(div_data_dir, "train_g_filterd.csv"))
train_p_filterd.to_csv(os.path.join(div_data_dir, "train_p_filterd.csv"))
test_g_filterd.to_csv(os.path.join(div_data_dir, "test_g_filterd.csv"))
test_p_filterd.to_csv(os.path.join(div_data_dir, "test_p_filterd.csv"))

# ====================================
# 重複列の削除
# ====================================

def remove_duplicate_columns(df):
    """
    DataFrameに全く同じデータの列が複数あれば、一番小さい名前の列だけを残す。

    Args:
        df: pandas DataFrame

    Returns:
        pandas DataFrame: 重複列が削除されたDataFrame
    """
    # 列名をソート
    sorted_cols = sorted(df.columns)

    # 重複列を格納するための辞書
    duplicate_cols = {}

    # ソートされた列名でループ
    for col in sorted_cols:
        # すでに重複列として登録されているか確認
        if col in duplicate_cols:
            continue  # 重複列はスキップ

        # 現在の列と比較する列
        for other_col in sorted_cols:
            # 同じ列は比較しない
            if col == other_col:
                continue

            # 現在の列がすでに重複列として登録されているかを確認
            if col in duplicate_cols:
                break

            # 比較対象が重複列として登録されているかを確認
            if other_col in duplicate_cols:
                continue

            # 列のデータが同じ場合
            if df[col].equals(df[other_col]):
                # 重複列として登録
                duplicate_cols[other_col] = col

    # 重複列を削除
    cols_to_keep = [col for col in sorted_cols if col not in duplicate_cols]
    df = df[cols_to_keep]

    return df

# 各データセットで重複列を削除
train_g_filt_col = remove_duplicate_columns(train_g_filterd)
train_p_filt_col = remove_duplicate_columns(train_p_filterd)
test_g_filt_col = remove_duplicate_columns(test_g_filterd)
test_p_filt_col = remove_duplicate_columns(test_p_filterd)

# 削除された列を確認
check_delete_columns(train_g_filterd, train_g_filt_col, 'train_g_filterd', 'train_g_filt_col')
check_delete_columns(train_p_filterd, train_p_filt_col, 'train_p_filterd', 'train_p_filt_col')
check_delete_columns(test_g_filterd, test_g_filt_col, 'test_g_filterd', 'test_g_filt_col')
check_delete_columns(test_p_filterd, test_p_filt_col, 'test_p_filterd', 'test_p_filt_col')

# 重複列削除済みデータを出力
train_g_filt_col.to_csv(os.path.join(div_data_dir, "train_g_filt_col.csv"))
train_p_filt_col.to_csv(os.path.join(div_data_dir, "train_p_filt_col.csv"))
test_g_filt_col.to_csv(os.path.join(div_data_dir, "test_g_filt_col.csv"))
test_p_filt_col.to_csv(os.path.join(div_data_dir, "test_p_filt_col.csv"))

# ====================================
# ユニーク値数=行数の列の特定と削除
# ====================================

def find_unique_count_equal_row_count_columns(df):
    """
    DataFrame型引数のdfにある列の中で、ユニークなデータの個数が行数と一致する列をリストアップして返す関数
    （つまり、各行で値が重複していない列＝ID的な列）

    Args:
        df: pandas DataFrame

    Returns:
        list: ユニークなデータの個数が行数と一致する列名のリスト
    """
    unique_count_equal_row_count_columns = []
    for col in df.columns:
        if df[col].nunique() == len(df):
            unique_count_equal_row_count_columns.append(col)
    return unique_count_equal_row_count_columns

# 各データセットでユニーク値数=行数の列を特定
uniq_cnt_eq_row_cnt_train_g = find_unique_count_equal_row_count_columns(train_g_filt_col)
uniq_cnt_eq_row_cnt_train_p = find_unique_count_equal_row_count_columns(train_p_filt_col)
uniq_cnt_eq_row_cnt_test_g = find_unique_count_equal_row_count_columns(test_g_filt_col)
uniq_cnt_eq_row_cnt_test_p = find_unique_count_equal_row_count_columns(test_p_filt_col)

print(f"uniq_cnt_eq_row_cnt_train_g : {uniq_cnt_eq_row_cnt_train_g}")
print(f"uniq_cnt_eq_row_cnt_train_p : {uniq_cnt_eq_row_cnt_train_p}")
print(f"uniq_cnt_eq_row_cnt_test_g : {uniq_cnt_eq_row_cnt_test_g}")
print(f"uniq_cnt_eq_row_cnt_test_p : {uniq_cnt_eq_row_cnt_test_p}")

# 削除対象列を決定（予測に不要なID列を削除、indexは残す）
drop_cols_train_g = uniq_cnt_eq_row_cnt_train_g  # 作業実績番号、受注番号_actual
drop_cols_train_p = uniq_cnt_eq_row_cnt_train_p  # 作業実績番号
drop_cols_test_g = list(set(uniq_cnt_eq_row_cnt_test_g) - set(['index']))  # index以外（予定組ラベルキー、受注番号）
drop_cols_test_p = list(set(uniq_cnt_eq_row_cnt_test_p) - set(['index']))  # index以外（予定組ラベルキー）

def remove_columns(df, drop_cols):
    """指定された列を削除する関数"""
    print(drop_cols)
    return df.drop(columns=drop_cols)

# ユニーク列を削除
train_g_uniq = remove_columns(train_g_filt_col, drop_cols_train_g)
train_p_uniq = remove_columns(train_p_filt_col, drop_cols_train_p)
test_g_uniq = remove_columns(test_g_filt_col, drop_cols_test_g)
test_p_uniq = remove_columns(test_p_filt_col, drop_cols_test_p)

# 削除結果を確認
check_delete_columns(train_g_filt_col, train_g_uniq, 'train_g_filt_col', 'train_g_uniq')
check_delete_columns(train_p_filt_col, train_p_uniq, 'train_p_filt_col', 'train_p_uniq')
check_delete_columns(test_g_filt_col, test_g_uniq, 'test_g_filt_col', 'test_g_uniq')
check_delete_columns(test_p_filt_col, test_p_uniq, 'test_p_filt_col', 'test_p_uniq')

# ユニーク列削除済みデータを出力
train_g_uniq.to_csv(os.path.join(div_data_dir, "train_g_uniq.csv"))
train_p_uniq.to_csv(os.path.join(div_data_dir, "train_p_uniq.csv"))
test_g_uniq.to_csv(os.path.join(div_data_dir, "test_g_uniq.csv"))
test_p_uniq.to_csv(os.path.join(div_data_dir, "test_p_uniq.csv"))

# ====================================
# trainのみにある列の特定と削除
# ====================================

def compare_columns(df_train, df_test):
    """
    2つのDataFrameの列を比較し、df_trainにあるがdf_testにない列をリストで返す。

    Args:
        df_train: 比較元のDataFrame。
        df_test: 比較対象のDataFrame。

    Returns:
        set: df_trainにあってdf_testにない列。
    """
    train_cols = set(df_train.columns)
    test_cols = set(df_test.columns)

    diff_cols = train_cols - test_cols

    print(diff_cols)

    return list(diff_cols)

# trainのみにある列を特定
diff_cols_g = compare_columns(train_g_uniq, test_g_uniq)
diff_cols_p = compare_columns(train_p_uniq, test_p_uniq)

print(len(diff_cols_g))  # グルアー用データでtrainのみにある列数
print(len(diff_cols_p))  # 印刷機用データでtrainのみにある列数

# trainのみにある列の中で、予測に必要な列を特定
# （時間系の列や新フォーマット判定列など、目的変数計算に必要な列）

# グルアー用データで保持する列
keep_col_g = [
    '残業時間',
    '付帯作業時間',
    '所要時間',
    '作成日',
    '合計時間',
    '作業時間',
    '計算_所要時間-(作業時間+残業時間)',
    '作業日_datetime',
    '終了時刻',
    '開始時刻',
    '新フォーマット',
]

# 印刷機用データで保持する列
keep_col_p = [
    '作業時間',
    '計算_所要時間-(作業時間+残業時間)',
    '作業日_datetime',
    '付帯作業時間',
    '開始時刻',
    '終了時刻',
    '所要時間',
    '残業時間',
    '作成日',
    '合計時間',
    '新フォーマット',
]

# 保持列リストに差異がないことを確認
print("保持列の差異:", set(keep_col_g) - set(keep_col_p))

# 削除対象列を決定（trainのみにある列から保持列を除いた列）
drop_cols_g = list(set(diff_cols_g) - set(keep_col_g))
drop_cols_p = list(set(diff_cols_p) - set(keep_col_p))

# 不要列を削除
train_g_f = remove_columns(train_g_uniq, drop_cols_g)
train_p_f = remove_columns(train_p_uniq, drop_cols_p)

# testデータからも不要列を削除（testでは号機コードのみ削除）
test_g_f = remove_columns(test_g_uniq, [])          # グルアーは削除なし
test_p_f = remove_columns(test_p_uniq, ["号機コード"])  # 印刷機は号機コードを削除

# 最終的なデータを出力
train_g_f.to_csv(os.path.join(div_data_dir, "train_g_f.csv"))
train_p_f.to_csv(os.path.join(div_data_dir, "train_p_f.csv"))
test_g_f.to_csv(os.path.join(div_data_dir, "test_g_f.csv"))
test_p_f.to_csv(os.path.join(div_data_dir, "test_p_f.csv"))

# ====================================
# データ確認用レポート生成（ydata-profiling）
# ====================================

# ydata-profilingを使用したデータ概要レポートの生成
# （メモリ使用量が大きいためコメントアウト）

# from ydata_profiling import ProfileReport
# import matplotlib
# matplotlib.rc('font', family='IPAPGothic') #全体のフォントをIPAゴシックに設定

# # 各データセットのプロファイルレポートを生成
# profile_train_g_f = ProfileReport(train_g_f,title="train_g_f.csv Profiling Report")
# profile_train_p_f = ProfileReport(train_p_f,title="train_p_f.csv Profiling Report")
# profile_test_g_f = ProfileReport(test_g_f,title="test_g_f.csv Profiling Report")
# profile_test_p_f = ProfileReport(test_p_f,title="test_p_f.csv Profiling Report")

# # レポート出力設定
# view_report_notebook = False # レポートをノートブックで表示する場合はTrue
# report_to_file = True # レポートをhtml出力る場合はTrue

# def ydata_report(profile, file_path):
#     """ydata-profilingレポートを出力する関数"""
#     if view_report_notebook:
#         profile.to_notebook_iframe()
#     if report_to_file:
#         profile.to_file(file_path)

# # 各データセットのレポートをHTMLファイルとして出力
# ydata_report(profile_train_g_f, os.path.join(div_data_dir, "train_g_f.html"))
# ydata_report(profile_train_p_f, os.path.join(div_data_dir, "train_p_f.html"))
# ydata_report(profile_test_g_f,  os.path.join(div_data_dir, "test_g_f.html"))
# ydata_report(profile_test_p_f,  os.path.join(div_data_dir, "test_p_f.html"))

print("データ前処理が完了しました。")
print("出力ファイル:")
print("- フィルタ済みデータ: eda_data/filtered_data/")
print("- 結合データ: eda_data/merge_data/, eda_data/merge_upper_data/")
print("- 分割・最終データ: eda_data/div_data/")
print("  - train_g_f.csv: グルアー（貼り機）用訓練データ")
print("  - train_p_f.csv: 印刷機用訓練データ")
print("  - test_g_f.csv: グルアー（貼り機）用テストデータ")
print("  - test_p_f.csv: 印刷機用テストデータ")
