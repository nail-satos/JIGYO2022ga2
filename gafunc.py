# 追加インストールしたライブラリ
import numpy as np
import pandas as pd 
# import streamlit as st
# import matplotlib.pyplot as plt 
# import japanize_matplotlib
# import seaborn as sns 

# ロゴの表示用
# from PIL import Image

# 標準ライブラリ
import random
import copy

# sns.set()
# japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def generate_0th_generation(operating_rate : int):

    # 全てゼロのデータフレームを作成
    zero = np.zeros((3,24), dtype=np.int8)
    df_shift = pd.DataFrame(zero, index=['マシンＡ', 'マシンＢ', 'マシンＣ'])

    # 初期状態では（とりあえず）フル稼働（Ａ=α、Ｂ=β、Ｃ=γ）
    df_shift.loc['マシンＡ'] = df_shift.loc['マシンＡ'].replace(0, 1)
    df_shift.loc['マシンＢ'] = df_shift.loc['マシンＢ'].replace(0, 2)
    df_shift.loc['マシンＣ'] = df_shift.loc['マシンＣ'].replace(0, 3)

    i = 0
    while i < 100:
        # さすがに偏り過ぎているので、製造部品をシャッフルする
        # ランダム：稼働させる機器の番号(0～2)
        c1 = random.randint(0, 2)
        r1 = random.randint(0, 23)
        c2 = random.randint(0, 2)
        r2 = random.randint(0, 23)

        temp = df_shift.iloc[c1, r1]
        df_shift.iloc[c1, r1] = df_shift.iloc[c2, r2]
        df_shift.iloc[c2, r2] = temp
        
        i = i + 1

    # 稼働率に準拠して、「遊休」の個数を算出する
    size = df_shift.size
    op_rate = operating_rate / 100
    idle_cnt = size - int(size * op_rate)

    # 決められた個数の遊休を挿入するループ
    i = 0
    while i < idle_cnt:
        c = random.randint(0, 2)
        r = random.randint(0, 23)
        if df_shift.iloc[c, r] != 0:
            df_shift.iloc[c, r] = 0
            i = i + 1

    return df_shift


### 部品の交換箇所をチェックして、2hの交換(9)を挿入する関数
def add_unit_switch(sr: pd.Series):

    new_shift = []
    unit_prev = sr.values.tolist()[0]   # 最初の部品を記録

    for unit in sr.values.tolist():
        if unit == unit_prev or unit == 0:
            # 記録していた前の部品と同じだったら...
            #（または、遊休中(0)の場合はノーカン）
            new_shift.append(unit)
        else:
            # 記録していた前の部品と違っていたら...
            # 2hの交換(9)を挿入する
            new_shift.append(9)
            new_shift.append(9)
            new_shift.append(unit)
            unit_prev = unit    # 部品を記録しなおし

    # 交換中を挿入したシフトの冒頭24hのみを戻す ※これで大丈夫なのか？
    return new_shift[0:24]


# 個体（データフレーム1個分）に対する評価を算出
def evaluation_individual(df_shift: pd.DataFrame, df_norma: pd.DataFrame, cap_params_list: list, co2_params_list: list, loss_list: list):

    # 作業用データフレームの作成
    df_remain = copy.deepcopy(df_norma)     # ノルマをコピーして、製造残を管理するデータフレームを作成
    df_co2    = copy.deepcopy(df_shift)     # ノルマをコピーして、CO2排出量を管理するデータフレームを作成
    df_co2    = df_co2.mask(df_remain != -1, 0)     # CO2排出量をオール0で初期化

    # データフレームのインデックスを振り直し（0～）
    df_shift = df_shift.reset_index(drop=True)


    print('\n df_shift')
    print(df_shift)

    # ペナルティをリストから復元
    incomplete_loss = loss_list[0]  # 生産不足のペナルティ
    complete_loss   = loss_list[1]  # 生産過多のペナルティ
    co2_loss        = loss_list[2]  # CO2排出量のペナルティ

    print('\n co2_params_list')
    print(co2_params_list)

    # 個体から1行ずつ取り出し（マシンＡ, Ｂ, Ｃ）
    for machine_no, row in df_shift.iterrows():

        # 1行（マシン）から時間帯ごとの状態（ステータス）を取り出し
        for hour, status in enumerate(row):

            # ステータスが1=部品α、2=部品β、3=部品γを作る
            if status == 1 or status ==2 or status ==3:

                parts_no = status - 1   # 添字の調整

                # 製造残から、製造した量を減算（その時、後ろの時間帯(h)に関しても、スライスで全て減算する）
                df_remain.iloc[parts_no, hour: ] = df_remain.iloc[parts_no, hour: ] - cap_params_list[machine_no][parts_no]

            # ステータスごとに添え字を設定
            if status == 1 or status ==2 or status ==3:
                status_idx = 0  # 製造時
            if status == 9:
                status_idx = 1  # 交換時
            if status == -1:
                status_idx = 2  # 整備時
            if status == 0:
                status_idx = 3  # 休止時

            # CO2排出量を加算
            df_co2.iloc[machine_no, hour] = df_co2.iloc[machine_no, hour] + co2_params_list[machine_no][status_idx]

    # 生産不足の算出：製造残が0以下（ノルマ以上は作れた）のものを0にする -> 残るのは生産不足のみとなる
    df_incomplete = df_remain.mask(df_remain <= 0, 0)     # maskの代わりにwhereを使うと挙動が逆になる
    incomplete_score = df_incomplete.sum().sum() * incomplete_loss * -1

    # 生産過多の算出：製造残が1以上（ノルマに達しなかった）のものを0にする -> 残るのは生産過多のみとなる
    df_complete = df_remain.mask(df_remain >= 1, 0)     # maskの代わりにwhereを使うと挙動が逆になる
    complete_score = df_complete.sum().sum() * complete_loss

    # CO2排出量スコアの算出
    co2_score = df_co2.sum().sum() * co2_loss * -1

    print('\n df_co2')
    print(df_co2)

    # 戻り値 = [生産不足スコア, 生産過多スコア, CO2排出量スコア]
    return [incomplete_score, complete_score, co2_score]


def generate_n_generation(df: pd.DataFrame):
    return(df)


