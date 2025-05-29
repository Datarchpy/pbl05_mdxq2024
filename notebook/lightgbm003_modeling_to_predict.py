# ライブラリのインストールとインポート
! pip install japanize_matplotlib
import importlib
import sys
import subprocess

# Google Colab 上で実行しているかどうかを判断するフラグ
ON_COLAB = "google.colab" in sys.modules
print(f"ON_COLAB: {ON_COLAB}")

# Google Colab環境での初期設定
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

# パス設定
import os

if ON_COLAB:
    # Google Colabを利用する場合
    path = '/content/drive/MyDrive/DXQuest/' # 適宜変更してください
else:
    # ローカルの場合
    path = "./" # 適宜変更してください
    #path = "/kaggle/input/pbl05-data/3-2_PBL05_data" # 適宜変更してください

# 必要なライブラリのインポート
import pandas as pd
import numpy as np
import japanize_matplotlib
pd.set_option('display.max_columns', None)

# 日本語フォント設定（コメントアウト済み）
# import os
# import importlib.util

# def japanize_ydata_profiling():
#     """ydata-profilingで日本語フォントを使用できるようにする設定関数"""
#     def add_font(file_path):
#         add_IPAexGothic = False
#
#         # ファイルを読み込む
#         with open(file_path, 'r', encoding='utf-8') as file:
#             lines = file.readlines()
#
#         # "font.sans-serif" のリスト内に "IPAexGothic" が存在するかチェック
#         ipaexgothic_present = False
#         for line in lines:
#             if '"font.sans-serif":' in line:
#                 if '"IPAexGothic"' in line:
#                     ipaexgothic_present = True
#                     break
#
#         # "IPAexGothic" が存在しなければ、リストの開始に追加
#         if not ipaexgothic_present:
#             for i, line in enumerate(lines):
#                 if '"font.sans-serif": [' in line:
#                     # [ の直後に "IPAexGothic", を追加
#                     insertion_point = line.find('[') + 1
#                     line = line[:insertion_point] + '"IPAexGothic", ' + line[insertion_point:]
#                     lines[i] = line
#                     add_IPAexGothic = True
#                     break
#
#         # 編集した内容でファイルを上書き保存
#         with open(file_path, 'w', encoding='utf-8') as file:
#             file.writelines(lines)
#
#         return add_IPAexGothic
#
#     # ydata_profiling のフォント設定を変更
#     library_path = importlib.util.find_spec("ydata_profiling").origin
#     library_dir = os.path.dirname(library_path)
#     file_path = library_dir + '/visualisation/context.py'
#     context_flg = add_font(file_path)
#
#     # seaborn のフォント設定を変更
#     library_path = importlib.util.find_spec("seaborn").origin
#     library_dir = os.path.dirname(library_path)
#     file_path = library_dir + '/rcmod.py'
#     rcmod_flg = add_font(file_path)
#
#     if context_flg or rcmod_flg:
#         raise ValueError("設定変更を反映するためにカーネルを再起動してください")
#     else:
#         print('japanize_ydata_profiling() is done')

# japanize_ydata_profiling()

# ディレクトリ作成のヘルパー関数
def makedir_helper(base_path: path, dir_name: str):
    """指定されたベースパスにディレクトリを作成し、そのパスを返す"""
    make_dir_path = os.path.join(base_path, dir_name)
    os.makedirs(make_dir_path, exist_ok=True)
    return make_dir_path

# 出力用ディレクトリの作成
fe_dir = makedir_helper(path, 'feature_engineering') # 特徴量エンジニアリングしたデータを出力するフォルダ
submit_dir = makedir_helper(path, 'submit') # 提出用ファイルを出力するフォルダ

# データ読み込み
dc_data_dir = os.path.join('/kaggle/input/pbl05-data-cleaning/', 'data_cleaning')

# 訓練データの読み込み
train_g_dc_csv = pd.read_csv(os.path.join(dc_data_dir, "train_g_dc.csv"))  # 訓練データ（一般印刷）
train_p_dc_csv = pd.read_csv(os.path.join(dc_data_dir, "train_p_dc.csv"))  # 訓練データ（パッケージ印刷）

# テストデータの読み込み
test_g_dc_csv = pd.read_csv(os.path.join(dc_data_dir, "test_g_dc.csv"))   # テストデータ（一般印刷）
test_p_dc_csv = pd.read_csv(os.path.join(dc_data_dir, "test_p_dc.csv"))   # テストデータ（パッケージ印刷）

# 可視化・機械学習ライブラリのインポート
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# 分布確認用のプロット関数（Tanakaさんのコードをコピー）
def plot_distribution(df, column_name):
    """指定された列の分布をボックスプロットとヒストグラムで可視化"""
    # ボックスプロット
    sns.boxplot(x = df[column_name])
    plt.title(f"Boxplot for {column_name}")
    plt.show()

    # ヒストグラム
    sns.histplot(df[column_name], bins = 30, kde = True)
    plt.title(f"Histogram for {column_name}")
    plt.xlabel(column_name)
    plt.ylabel("Frequency")
    plt.show()

# パッケージ印刷データの前処理
# 作業時間が0以下のデータを除外（異常値除去）
train_p_dc_csv = train_p_dc_csv[(train_p_dc_csv["fix付帯作業時間"] > 0) & (train_p_dc_csv["作業時間"] > 0)]

# 元データの分布確認
plot_distribution(train_p_dc_csv, "作業時間")
plot_distribution(train_p_dc_csv, "fix付帯作業時間")

# Box-Cox変換とその逆変換の関数定義
from scipy import stats

def boxcox_transform(data, lmbda=None):
    """
    Box-Cox変換を行う関数
    
    Args:
        data: 変換対象のデータ（NumPy配列またはPandas Series）
        lmbda: Box-Cox変換のパラメータ。Noneの場合は最適なλを推定する
    
    Returns:
        変換後のデータと推定されたλのタプル
    """
    transformed_data, lmbda_estimated = stats.boxcox(data)
    return transformed_data, lmbda_estimated

def inverse_boxcox_transform(data, lmbda):
    """
    Box-Cox逆変換を行う関数
    
    Args:
        data: 逆変換対象のデータ（NumPy配列またはPandas Series）
        lmbda: Box-Cox変換に使用したλ
    
    Returns:
        逆変換後のデータ
    """
    if lmbda == 0:
        return np.exp(data)
    else:
        return np.exp(np.log(lmbda * data + 1) / lmbda)

# パッケージ印刷データにBox-Cox変換を適用（正規分布に近づける）
train_p_dc_csv['作業時間_boxcox'], lmbda_p_nwt = boxcox_transform(train_p_dc_csv['作業時間'])
train_p_dc_csv['fix付帯作業時間_boxcox'], lmbda_p_awt = boxcox_transform(train_p_dc_csv['fix付帯作業時間'])

# 変換後の分布確認
plot_distribution(train_p_dc_csv, "作業時間_boxcox")
plot_distribution(train_p_dc_csv, "fix付帯作業時間_boxcox")

# パッケージ印刷データの特徴量エンジニアリング
# 色数関連の特徴量作成
train_p_dc_csv["色数(一般+特殊)"] = train_p_dc_csv["色数(一般)"] + train_p_dc_csv["色数(特殊)"]
test_p_dc_csv["色数(一般+特殊)"] = test_p_dc_csv["色数(一般)"] + test_p_dc_csv["色数(特殊)"]

train_p_dc_csv["表+裏色数"] = train_p_dc_csv["表色数"] + train_p_dc_csv["裏色数"]
test_p_dc_csv["表+裏色数"] = test_p_dc_csv["表色数"] + test_p_dc_csv["裏色数"]

# 作成した特徴量の確認
train_p_dc_csv[['色数(一般)', '色数(特殊)', '表色数', '裏色数', "色数(一般+特殊)", "表+裏色数"]]
test_p_dc_csv[['色数(一般)', '色数(特殊)', '表色数', '裏色数', "色数(一般+特殊)", "表+裏色数"]]

# 欠損値の確認と処理
# 一般色数関連の欠損値確認
train_p_dc_csv["色数(一般)"].isna().sum(), train_p_dc_csv["色数(特殊)"].isna().sum(), train_p_dc_csv["色数(一般+特殊)"].isna().sum()
test_p_dc_csv["色数(一般)"].isna().sum(), test_p_dc_csv["色数(特殊)"].isna().sum(), test_p_dc_csv["色数(一般+特殊)"].isna().sum()

# 表・裏色数関連の欠損値確認
train_p_dc_csv["表色数"].isna().sum(), train_p_dc_csv["裏色数"].isna().sum(), train_p_dc_csv["表+裏色数"].isna().sum()
test_p_dc_csv["表色数"].isna().sum(), test_p_dc_csv["裏色数"].isna().sum(), test_p_dc_csv["表+裏色数"].isna().sum()

# 欠損値の詳細確認
train_p_dc_csv[train_p_dc_csv["色数(一般)"].isna()]

# 色数(一般)が欠損している場合の色数(特殊)を0に設定
train_p_dc_csv.loc[train_p_dc_csv["色数(一般)"].isna(), "色数(特殊)"] = 0
test_p_dc_csv.loc[test_p_dc_csv["色数(一般)"].isna(), "色数(特殊)"] = 0

# 欠損値の手動補完（業務知識に基づく）
train_p_dc_csv.loc[train_p_dc_csv["index"] == 3118, "色数(一般)"] = train_p_dc_csv.loc[train_p_dc_csv["index"] == 3118, "表色数"] # 裏はNaNなので足さない
train_p_dc_csv.loc[train_p_dc_csv["index"] == 7626, "色数(一般)"] = train_p_dc_csv.loc[train_p_dc_csv["index"] == 7626, "表色数"] + train_p_dc_csv.loc[train_p_dc_csv["index"] == 7626, "裏色数"]
train_p_dc_csv.loc[train_p_dc_csv["index"] == 8294, "色数(一般)"] = train_p_dc_csv.loc[train_p_dc_csv["index"] == 8294, "表色数"] + train_p_dc_csv.loc[train_p_dc_csv["index"] == 8294, "裏色数"]
train_p_dc_csv.loc[train_p_dc_csv["index"] == 8302, "色数(一般)"] = train_p_dc_csv.loc[train_p_dc_csv["index"] == 8302, "表色数"] + train_p_dc_csv.loc[train_p_dc_csv["index"] == 8302, "裏色数"]

# 補完後の欠損値確認
train_p_dc_csv[train_p_dc_csv["色数(一般+特殊)"].isna()]
train_p_dc_csv["色数(一般)"].isna().sum(), train_p_dc_csv["色数(特殊)"].isna().sum(), train_p_dc_csv["色数(一般+特殊)"].isna().sum()
test_p_dc_csv["色数(一般)"].isna().sum(), test_p_dc_csv["色数(特殊)"].isna().sum(), test_p_dc_csv["色数(一般+特殊)"].isna().sum()

# 特徴量の再計算
train_p_dc_csv["色数(一般+特殊)"] = train_p_dc_csv["色数(一般)"] + train_p_dc_csv["色数(特殊)"]
test_p_dc_csv["色数(一般+特殊)"] = test_p_dc_csv["色数(一般)"] + test_p_dc_csv["色数(特殊)"]

# 裏色数の欠損値処理
train_p_dc_csv.loc[train_p_dc_csv["表色数"].isna(), "裏色数"]
test_p_dc_csv.loc[test_p_dc_csv["表色数"].isna(), "裏色数"]

# 裏色数がNaNな行の裏色数を0にする（片面印刷の場合）
train_p_dc_csv.loc[train_p_dc_csv["裏色数"].isna(), "裏色数"] = 0
test_p_dc_csv.loc[test_p_dc_csv["裏色数"].isna(), "裏色数"] = 0

# 表+裏色数の再計算
train_p_dc_csv["表+裏色数"] = train_p_dc_csv["表色数"] + train_p_dc_csv["裏色数"]
test_p_dc_csv["表+裏色数"] = test_p_dc_csv["表色数"] + test_p_dc_csv["裏色数"]

# 最終的な欠損値確認
train_p_dc_csv["表色数"].isna().sum(), train_p_dc_csv["裏色数"].isna().sum(), train_p_dc_csv["表+裏色数"].isna().sum()
test_p_dc_csv["表色数"].isna().sum(), test_p_dc_csv["裏色数"].isna().sum(), test_p_dc_csv["表+裏色数"].isna().sum()

# 残りの欠損値処理
train_p_dc_csv[train_p_dc_csv["表+裏色数"].isna()]
test_p_dc_csv[test_p_dc_csv["表+裏色数"].isna()]

# 表色数の欠損値を他の色数情報から推定
train_p_dc_csv.loc[train_p_dc_csv["表色数"].isna(), "表色数"] = train_p_dc_csv.loc[train_p_dc_csv["表色数"].isna(), "色数(一般+特殊)"] - train_p_dc_csv.loc[train_p_dc_csv["表色数"].isna(), "裏色数"]
test_p_dc_csv.loc[test_p_dc_csv["表色数"].isna(), "表色数"] = test_p_dc_csv.loc[test_p_dc_csv["表色数"].isna(), "色数(一般+特殊)"] - test_p_dc_csv.loc[test_p_dc_csv["表色数"].isna(), "裏色数"]

# 最終的な表+裏色数の計算
train_p_dc_csv["表+裏色数"] = train_p_dc_csv["表色数"] + train_p_dc_csv["裏色数"]
test_p_dc_csv["表+裏色数"] = test_p_dc_csv["表色数"] + test_p_dc_csv["裏色数"]

# 色数の整合性確認
(train_p_dc_csv["色数(一般+特殊)"] - train_p_dc_csv["表+裏色数"]).value_counts()

# 両面印刷フラグの作成
train_p_dc_csv["両面印刷"] = 0
train_p_dc_csv.loc[train_p_dc_csv["裏色数"] != 0, "両面印刷"] = 1

test_p_dc_csv["両面印刷"] = 0
test_p_dc_csv.loc[test_p_dc_csv["裏色数"] != 0, "両面印刷"] = 1

# 寸法関連の特徴量作成（面積計算）
train_p_dc_csv['刷本寸法幅*刷本寸法長さ'] = train_p_dc_csv["刷本寸法幅"] * train_p_dc_csv["刷本寸法長さ"]
train_p_dc_csv['展開寸法幅*展開寸法長さ'] = train_p_dc_csv["展開寸法幅"] * train_p_dc_csv["展開寸法長さ"]

test_p_dc_csv['刷本寸法幅*刷本寸法長さ'] = test_p_dc_csv["刷本寸法幅"] * test_p_dc_csv["刷本寸法長さ"]
test_p_dc_csv['展開寸法幅*展開寸法長さ'] = test_p_dc_csv["展開寸法幅"] * test_p_dc_csv["展開寸法長さ"]

# 予備数関連の特徴量作成
train_p_dc_csv['加工予備数+印刷予備数'] = train_p_dc_csv["加工予備数"] * train_p_dc_csv["印刷予備数"]
test_p_dc_csv['加工予備数+印刷予備数'] = test_p_dc_csv["加工予備数"] * test_p_dc_csv["印刷予備数"]

# 比率関連の特徴量作成
# 通し数 対 加工予備数の比率
train_p_dc_csv['通し数/加工予備数'] = train_p_dc_csv["通し数"] / train_p_dc_csv["加工予備数"]
test_p_dc_csv['通し数/加工予備数'] = test_p_dc_csv["通し数"] / test_p_dc_csv["加工予備数"]

# 通し数 対 印刷予備数の比率
train_p_dc_csv['通し数/印刷予備数'] = train_p_dc_csv["通し数"] / train_p_dc_csv["印刷予備数"]
test_p_dc_csv['通し数/印刷予備数'] = test_p_dc_csv["通し数"] / test_p_dc_csv["印刷予備数"]

# 通し数 対 加工予備数+印刷予備数の比率
train_p_dc_csv['通し数/(加工予備数+印刷予備数)'] = train_p_dc_csv["通し数"] / train_p_dc_csv["加工予備数+印刷予備数"]
test_p_dc_csv['通し数/(加工予備数+印刷予備数)'] = test_p_dc_csv["通し数"] / test_p_dc_csv["加工予備数+印刷予備数"]

# 一般印刷データの前処理
# 作業時間が0以下のデータを除外（異常値除去）
train_g_dc_csv = train_g_dc_csv[(train_g_dc_csv["fix付帯作業時間"] > 0) & (train_g_dc_csv["作業時間"] > 0)]
train_g_dc_csv.shape

# 一般印刷データの分布確認
plot_distribution(train_g_dc_csv, "作業時間")
plot_distribution(train_g_dc_csv, "fix付帯作業時間")

# 一般印刷データにBox-Cox変換を適用
train_g_dc_csv['作業時間_boxcox'], lmbda_g_nwt = boxcox_transform(train_g_dc_csv['作業時間'])
train_g_dc_csv['fix付帯作業時間_boxcox'], lmbda_g_awt = boxcox_transform(train_g_dc_csv['fix付帯作業時間'])

# 変換後の分布確認
plot_distribution(train_g_dc_csv, "作業時間_boxcox")
plot_distribution(train_g_dc_csv, "fix付帯作業時間_boxcox")

# 一般印刷データの特徴量エンジニアリング
# 展開寸法関連の特徴量作成（周囲長の計算など）
train_g_dc_csv["展開寸法長さ+(展開寸法幅*2)"] = train_g_dc_csv["展開寸法長さ"] + (train_g_dc_csv["展開寸法幅"] * 2)
test_g_dc_csv["展開寸法長さ+(展開寸法幅*2)"] = test_g_dc_csv["展開寸法長さ"] + (test_g_dc_csv["展開寸法幅"] * 2)

train_g_dc_csv["(展開寸法長さ*2)+展開寸法幅"] = (train_g_dc_csv["展開寸法長さ"] * 2) + train_g_dc_csv["展開寸法幅"]
test_g_dc_csv["(展開寸法長さ*2)+展開寸法幅"] = (test_g_dc_csv["展開寸法長さ"] * 2) + test_g_dc_csv["展開寸法幅"]

train_g_dc_csv["展開寸法長さ+展開寸法幅"] = train_g_dc_csv["展開寸法長さ"] + train_g_dc_csv["展開寸法幅"]
test_g_dc_csv["展開寸法長さ+展開寸法幅"] = test_g_dc_csv["展開寸法長さ"] + test_g_dc_csv["展開寸法幅"]

# 比率特徴量の作成
train_g_dc_csv["枚数/予備数量"] = train_g_dc_csv["枚数"] / train_g_dc_csv["予備数量"]
test_g_dc_csv["枚数/予備数量"] = test_g_dc_csv["枚数"] / test_g_dc_csv["予備数量"]

# データ型変換のヘルパー関数
def object2category(df, column_name):
    """object型をcategory型に変換する"""
    if column_name not in df.columns:
        raise ValueError(f"{column_name} is not in DataFrame")
    if df[column_name].dtype == "category":
        return df
    if df[column_name].dtype != "object":
        raise ValueError(f"{column_name} is not object type")
    df_mod = df.copy()
    df_mod[column_name] = df[column_name].astype("category")
    return df_mod

# パッケージ印刷用の特徴量とターゲット変数の準備
train_p_dc_csv.columns

# パッケージ印刷で使用する特徴量リスト
p_feature_coluns = [
    '粘着タイプ', '用紙種別', '流用受注有無',
    '刷了数', '刷本寸法幅', '刷本寸法長さ',
    '加工予備数', '印刷予備数', '台数', '号機名', '展開寸法幅',
    '展開寸法長さ', '通し数', '色数(一般+特殊)', '両面印刷',
    '通し実数', '連量', '部品区分', '部品区分名',
    '表色数', '裏色数', '色数(一般)', '色数(特殊)',
    '刷本寸法幅*刷本寸法長さ', '展開寸法幅*展開寸法長さ',
    '加工予備数+印刷予備数', '通し数/加工予備数', '通し数/印刷予備数', '通し数/(加工予備数+印刷予備数)',
]

# パッケージ印刷データの特徴量とターゲット変数の分離
train_p_feature = train_p_dc_csv[p_feature_coluns]
train_p_tgt_nwt = train_p_dc_csv["作業時間_boxcox"]          # 作業時間（Box-Cox変換済み）
train_p_tgt_awt = train_p_dc_csv["fix付帯作業時間_boxcox"]   # 付帯作業時間（Box-Cox変換済み）

test_p_feature = test_p_dc_csv[p_feature_coluns]

# データ型の確認
train_p_tgt_nwt.info()
train_p_tgt_awt.info()
train_p_feature.info()
test_p_feature.info()

# パッケージ印刷データのカテゴリ変数をcategory型に変換
train_p_feature = object2category(train_p_feature, "粘着タイプ")
train_p_feature = object2category(train_p_feature, "用紙種別")
train_p_feature = object2category(train_p_feature, "号機名")
train_p_feature = object2category(train_p_feature, "部品区分名")

test_p_feature = object2category(test_p_feature, "粘着タイプ")
test_p_feature = object2category(test_p_feature, "用紙種別")
test_p_feature = object2category(test_p_feature, "号機名")
test_p_feature = object2category(test_p_feature, "部品区分名")

# 変換後のデータ型確認
train_p_feature.info()
test_p_feature.info()

# 一般印刷用の特徴量とターゲット変数の準備
train_g_dc_csv.columns

# 一般印刷で使用する特徴量リスト
g_feature_coluns = [
    '用紙種別', '予備数量', '流用受注有無',
    '仕上数量','加工数量', '合計数量',
    '展開寸法幅', '展開寸法長さ', '枚数', '連量',
    '展開寸法長さ+(展開寸法幅*2)', '(展開寸法長さ*2)+展開寸法幅', '展開寸法長さ+展開寸法幅',
]

# 一般印刷データの特徴量とターゲット変数の分離
train_g_feature = train_g_dc_csv[g_feature_coluns]
train_g_tgt_nwt = train_g_dc_csv["作業時間_boxcox"]          # 作業時間（Box-Cox変換済み）
train_g_tgt_awt = train_g_dc_csv["fix付帯作業時間_boxcox"]   # 付帯作業時間（Box-Cox変換済み）

test_g_feature = test_g_dc_csv[g_feature_coluns]

# データ型の確認
train_g_feature.info()
test_g_feature.info()

# 一般印刷データのカテゴリ変数をcategory型に変換
train_g_feature = object2category(train_g_feature, "用紙種別")
test_g_feature = object2category(test_g_feature, "用紙種別")

# 変換後のデータ型確認
train_g_feature.info()
test_g_feature.info()

# カテゴリ変数の列名を取得するヘルパー関数
def get_categorical_columns(df):
    """データフレームからカテゴリ型の列名を取得"""
    categorical_columns = [col for col in df.columns if df[col].dtype == "category"]
    return categorical_columns

# パッケージ印刷データのカテゴリ変数確認
train_p_cat_col = get_categorical_columns(train_p_feature)
train_p_cat_col

# 一般印刷データのカテゴリ変数確認
train_g_cat_col = get_categorical_columns(train_g_feature)
train_g_cat_col

# ラベルエンコーディング関数
def label_encode(train_df, test_df, column_name):
    """
    指定された列に対してラベルエンコーディングを実行
    訓練データとテストデータで一貫性を保つ
    """
    if column_name not in train_df.columns:
        raise ValueError(f"{column_name} is not in train DataFrame")
    if column_name not in test_df.columns:
        raise ValueError(f"{column_name} is not in test DataFrame")

    label_encoder = LabelEncoder()
    train = train_df.copy()
    test = test_df.copy()

    # 訓練データでエンコーダーを学習し、両データに適用
    train[column_name] = label_encoder.fit_transform(train[column_name])
    test[column_name] = label_encoder.transform(test[column_name])

    # エンコード後はcategory型に変換
    train[column_name] = train[column_name].astype("category")
    test[column_name] = test[column_name].astype("category")

    return train, test

# 一般印刷データの用紙種別にラベルエンコーディングを適用
train_g_feature, test_g_feature = label_encode(train_g_feature, test_g_feature, "用紙種別")

# ペアプロット描画関数
def pair_plot(train_X, train_y):
    """
    特徴量間の相関関係を可視化するペアプロットを作成
    
    Args:
        train_X: 特徴量データ
        train_y: ターゲット変数
    """
    if "y" in train_X.columns:
        raise ValueError("すでにy列がDataFrame train_Xに含まれています")
    
    # ターゲット変数を追加してペアプロットを描画
    data = train_X.copy()
    data["y"] = train_y
    sns.pairplot(data)
    plt.show()

# 機械学習モデリング関数
def modeling(train_X, train_y, test_X, stratified_col, categorical_col, n_splits):
    """
    LightGBMを使用したクロスバリデーション付きモデリング
    
    Args:
        train_X: 訓練用特徴量データ
        train_y: 訓練用ターゲット変数
        test_X: テスト用特徴量データ  
        stratified_col: 層化分割に使用する列名
        categorical_col: カテゴリ変数の列名リスト
        n_splits: クロスバリデーションの分割数
    
    Returns:
        predictions: テストデータの予測値
        feature_importances: 特徴量重要度
    """
    # 入力データの検証
    if stratified_col not in train_X.columns:
        raise ValueError(f"train_Xに{stratified_col}列が含まれていません")
    if train_X[stratified_col].dtype != "category":
        raise ValueError(f"{stratified_col}列がcategory型ではありません")
    if "y" in train_X.columns:
        raise ValueError("すでにy列がDataFrame train_Xに含まれています")

    # データの準備
    data = train_X.copy()
    data["y"] = train_y

    # 層化K分割クロスバリデーションの設定
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # モデルと特徴量重要度を保存するリスト・データフレーム
    models = []
    feature_importances = pd.DataFrame()
    feature_importances["Feature"] = train_X.columns

    # クロスバリデーションのループ
    for fold, (train_index, valid_index) in enumerate(skf.split(data, data[stratified_col])):
        # 訓練・検証データの分割
        train_fold = data.iloc[train_index]
        valid_fold = data.iloc[valid_index]

        train_X = train_fold.drop("y", axis=1)
        train_y = train_fold["y"]
        valid_X = valid_fold.drop("y", axis=1)
        valid_y = valid_fold["y"]

        # LightGBM用のデータセット作成
        lgb_train = lgb.Dataset(train_X, train_y)
        lgb_eval = lgb.Dataset(valid_X, valid_y, reference=lgb_train)

        # LightGBMのパラメータ設定
        params = {
            "objective": "regression",  # 回帰問題
            "metric": "mae",           # 平均絶対誤差を評価指標とする
            "random_state": 42,        # 再現性のための乱数シード
        }
        
        # モデルの訓練
        model = lgb.train(
            params,
            lgb_train,
            valid_sets=[lgb_train, lgb_eval],
            categorical_feature=categorical_col,  # カテゴリ変数の指定
        )
        models.append(model)

        # 特徴量重要度を保存（gainベースの重要度）
        fold_importance = model.feature_importance(importance_type="gain")
        feature_importances[f"Fold_{fold + 1}"] = fold_importance

    # 各foldの平均重要度を計算
    feature_importances["Average"] = feature_importances.iloc[:, 1:].mean(axis=1)

    # 特徴量重要度をプロット
    feature_importances = feature_importances.sort_values(by="Average", ascending=False)
    plt.figure(figsize=(10, 6))
    plt.barh(feature_importances["Feature"], feature_importances["Average"])
    plt.xlabel("Feature Importance (Average Gain)")
    plt.ylabel("Feature")
    plt.title("Feature Importances")
    plt.gca().invert_yaxis()  # Y軸を反転して重要度の高い順に表示
    plt.show()

    # テストデータの予測（全モデルの平均を取る）
    predictions = np.mean([model.predict(test_X) for model in models], axis=0)

    return predictions, feature_importances

# パッケージ印刷データのモデリング実行
# 作業時間の予測
pred_p_nwt, fi_p_nwt = modeling(train_p_feature, train_p_tgt_nwt, test_p_feature, "号機名", train_p_cat_col, 5)
# Box-Cox変換の逆変換を適用して元のスケールに戻す
pred_p_nwt = inverse_boxcox_transform(pred_p_nwt, lmbda_p_nwt)

# 付帯作業時間の予測
pred_p_awt, fi_p_awt = modeling(train_p_feature, train_p_tgt_awt, test_p_feature, "号機名", train_p_cat_col, 5)
# Box-Cox変換の逆変換を適用して元のスケールに戻す
pred_p_awt = inverse_boxcox_transform(pred_p_awt, lmbda_p_awt)

# 一般印刷データのモデリング実行
# 作業時間の予測
pred_g_nwt, fi_g_nwt = modeling(train_g_feature, train_g_tgt_nwt, test_g_feature, "用紙種別", train_g_cat_col, 5)
# Box-Cox変換の逆変換を適用して元のスケールに戻す
pred_g_nwt = inverse_boxcox_transform(pred_g_nwt, lmbda_g_nwt)

# 一般印刷の作業時間予測値を5の倍数で丸める（業務要件による）
pred_g_nwt = np.round(pred_g_nwt / 5) * 5
# 代替案：2.5の倍数で丸める場合
# pred_g_nwt = np.round((pred_g_nwt * 2) / 5) * 2.5

# 付帯作業時間の予測
pred_g_awt, fi_g_awt = modeling(train_g_feature, train_g_tgt_awt, test_g_feature, "用紙種別", train_g_cat_col, 5)
# Box-Cox変換の逆変換を適用して元のスケールに戻す
pred_g_awt = inverse_boxcox_transform(pred_g_awt, lmbda_g_awt)

# 一般印刷の付帯作業時間予測値を5の倍数で丸める（業務要件による）
pred_g_awt = np.round(pred_g_awt / 5) * 5
# 代替案：2.5の倍数で丸める場合
# pred_g_awt = np.round((pred_g_awt * 2) / 5) * 2.5

# 予測結果の確認
len(pred_p_nwt)
len(pred_p_awt)
test_p_dc_csv.shape

# パッケージ印刷の予測結果をデータフレームに整理
pred_p_df = pd.DataFrame({
    "index": test_p_dc_csv["index"], 
    "作業時間": pred_p_nwt, 
    "fix付帯作業時間": pred_p_awt
})
pred_p_df.shape
pred_p_df

# 一般印刷の予測結果の確認
len(pred_g_nwt)
len(pred_g_nwt)
test_g_dc_csv.shape

# 一般印刷の予測結果をデータフレームに整理
pred_g_df = pd.DataFrame({
    "index": test_g_dc_csv["index"], 
    "作業時間": pred_g_nwt, 
    "fix付帯作業時間": pred_g_awt
})
pred_g_df.shape
pred_g_df

# パッケージ印刷と一般印刷の予測結果を結合
pred_df = pd.concat([pred_p_df, pred_g_df])
pred_df

# インデックス順にソート（提出形式に合わせる）
sorted_pred_df = pred_df.sort_values("index")
sorted_pred_df

# 提出ファイルの作成
from datetime import datetime, timedelta

# 現在時刻を取得してJST（日本標準時）に変換
now = datetime.now()  # グリニッジ標準時として取得
jst_now = now + timedelta(hours=9)  # 9時間加算して日本の標準時間に

# タイムスタンプ付きファイル名を生成
timestamp_str = jst_now.strftime("%Y%m%d_%H%M%S")  # 変換後の日時でフォーマット
filename = f"submission_{timestamp_str}.csv"
print(filename)

# 提出用CSVファイルの保存（ヘッダーなし、インデックスなし）
sorted_pred_df.to_csv(os.path.join(submit_dir, filename), index=False, header=False)
