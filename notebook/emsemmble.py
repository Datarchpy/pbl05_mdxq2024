import pandas as pd

# =======================================
# 1. 予測結果ファイルの読み込み
# =======================================
# 各モデルの予測結果をヘッダーなしで読み込み
# 通常、予測結果ファイルは [index, 正味作業時間, 付帯作業時間] の形式

# file1: メインモデルの予測結果（最も高い重みを設定予定）
file1 = pd.read_csv('/kaggle/input/steelpipe-submission/submission_20241215_161821.csv', header=None)

# file2: サブモデル1の予測結果（中程度の重みを設定予定）
file2 = pd.read_csv('/kaggle/input/s003-2submission-2/s003submission (2).csv', header=None)

# file3: サブモデル2の予測結果（最も低い重みを設定予定）
file3 = pd.read_csv('/kaggle/input/ag014-submission/submission (4).csv', header=None)

# =======================================
# 2. アンサンブル重みの設定
# =======================================
# 重み付きアンサンブルのための重み設定
# 重みの合計は1.0になるように設定（0.6 + 0.3 + 0.1 = 1.0）

weight1 = 0.6  # file1（メインモデル）の重み - 最も信頼性が高いモデル
weight2 = 0.3  # file2（サブモデル1）の重み - 中程度の信頼性
weight3 = 0.1  # file3（サブモデル2）の重み - 補完的な役割

print(f"アンサンブル重み設定:")
print(f"  モデル1: {weight1} (メインモデル)")
print(f"  モデル2: {weight2} (サブモデル1)")  
print(f"  モデル3: {weight3} (サブモデル2)")
print(f"  重みの合計: {weight1 + weight2 + weight3}")

# =======================================
# 3. データ整合性チェック
# =======================================
# 各ファイルの行数が同じであることを確認
# アンサンブルを行うためには、全てのファイルが同じ数の予測結果を持つ必要がある

print(f"\n各ファイルの行数確認:")
print(f"  file1: {len(file1)}行")
print(f"  file2: {len(file2)}行") 
print(f"  file3: {len(file3)}行")

# 行数が異なる場合はエラーを発生させて処理を停止
if len(file1) != len(file2) or len(file1) != len(file3):
    raise ValueError("ファイルの行数が異なります。同じ行数のファイルを使用してください。")

print("✓ 全ファイルの行数が一致しています")

# =======================================
# 4. 重み付きアンサンブルの実行
# =======================================
# 新しいDataFrameを作成してアンサンブル結果を格納

ensemble = pd.DataFrame()

# 列0: インデックス（識別子）をそのまま使用
# 通常、この列は全てのファイルで同じ値を持つはず
ensemble[0] = file1[0]

# 列1: 正味作業時間の重み付きアンサンブル
# 各モデルの予測値に重みを掛けて合計
ensemble[1] = (file1[1] * weight1 + 
               file2[1] * weight2 + 
               file3[1] * weight3)

# 列2: 付帯作業時間の重み付きアンサンブル  
# 各モデルの予測値に重みを掛けて合計
ensemble[2] = (file1[2] * weight1 + 
               file2[2] * weight2 + 
               file3[2] * weight3)

print(f"\n✓ 重み付きアンサンブル完了")
print(f"  アンサンブル結果の行数: {len(ensemble)}行")

# =======================================
# 5. アンサンブル結果の統計情報表示
# =======================================
print(f"\n=== アンサンブル結果の統計情報 ===")
print(f"正味作業時間（列1）の統計:")
print(f"  平均: {ensemble[1].mean():.2f}")
print(f"  最小: {ensemble[1].min():.2f}")
print(f"  最大: {ensemble[1].max():.2f}")
print(f"  標準偏差: {ensemble[1].std():.2f}")

print(f"\n付帯作業時間（列2）の統計:")
print(f"  平均: {ensemble[2].mean():.2f}")
print(f"  最小: {ensemble[2].min():.2f}")
print(f"  最大: {ensemble[2].max():.2f}")
print(f"  標準偏差: {ensemble[2].std():.2f}")

# =======================================
# 6. 結果の保存
# =======================================
# インデックスをリセットして整理
ensemble = ensemble.reset_index(drop=True)

# アンサンブル結果をCSVファイルとして保存
# index=False: 行番号を保存しない
# header=False: 列名を保存しない（提出形式に合わせる）
output_filename = 'ensemble011_predict.csv'
ensemble.to_csv(output_filename, index=False, header=False)

print(f"\n✓ 重み付きアンサンブルした予測結果を '{output_filename}' に保存しました")

# =======================================
# 7. 保存結果の確認
# =======================================
# 保存されたファイルの最初の数行を表示して確認
print(f"\n=== 保存された結果のサンプル（先頭5行） ===")
print(ensemble.head())

print(f"\n=== アンサンブル処理完了 ===")
print(f"ファイル名: {output_filename}")
print(f"データ形式: [インデックス, 正味作業時間予測, 付帯作業時間予測]")

# =======================================
# 補足情報
# =======================================
# アンサンブル学習の利点:
# 1. 個別モデルの予測誤差を相互に補完
# 2. 過学習のリスクを軽減  
# 3. より安定した予測性能を実現
# 4. 異なるアルゴリズムの強みを組み合わせ
