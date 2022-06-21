""" streamlit_demo
streamlitでIrisデータセットの分析結果を Web アプリ化するモジュール

【Streamlit 入門 1】Streamlit で機械学習のデモアプリ作成 – DogsCox's tech. blog
https://dogscox-trivial-tech-blog.com/posts/streamlit_demo_iris_decisiontree/
"""

# 追加インストールしたライブラリ
import numpy as np
import pandas as pd 
import streamlit as st
import matplotlib.pyplot as plt 
import japanize_matplotlib
import seaborn as sns 

# ロゴの表示用
from PIL import Image

# 標準ライブラリ
import random
import copy

# 自作パッケージ
# from gafunc import display_table 
# from gafunc import display_individual 
from gafunc import generate_0th_generation 
from gafunc import add_unit_switch 
from gafunc import evaluation_individual 
from gafunc import generate_n_generation 


sns.set()
japanize_matplotlib.japanize()  # 日本語フォントの設定

# matplotlib / seaborn の日本語の文字化けを直す、汎用的かつ一番簡単な設定方法 | BOUL
# https://boul.tech/mplsns-ja/


def display_table(title, df: pd.DataFrame):
    st.subheader(title)
    st.table(df)


def display_individual(title, df: pd.DataFrame, score_list: list):

    # データフレームを表示
    st.subheader(title)
    st.text(f'生産不足：{score_list[0]} + 生産過多：{score_list[1]} + CO2排出量：{score_list[2]} = 合計：{score_list[0] + score_list[1] + score_list[2]} 点')
    st.table(df)

    # Streamlitでdataframeを表示させる | ITブログ
    # https://kajiblo.com/streamlit-dataframe/

    
def main():
    """ メインモジュール
    """

    # セッションステートを初期化する
    if 'ini_flg' not in st.session_state:

        st.session_state.ini_flg = True

        # パラーメータの初期設定（CO2排出量）
        temp = []
        # temp.append([10,10,10,10])   # マシンA（製造時、交換時、整備時、遊休時）
        # temp.append([ 3, 3, 3, 3])   # マシンB（製造時、交換時、整備時、遊休時）
        # temp.append([ 1, 1, 1, 1])   # マシンC（製造時、交換時、整備時、遊休時）
        temp.append([10, 7, 5, 3])   # マシンA（製造時、交換時、整備時、遊休時）
        temp.append([ 5, 4, 3, 2])   # マシンB（製造時、交換時、整備時、遊休時）
        temp.append([ 3, 2, 1, 1])   # マシンC（製造時、交換時、整備時、遊休時）
        st.session_state.co2_params_list = temp

        # パラーメータの初期設定（製造能力:キャパシティ）
        temp = []
        temp.append([10, 10,  5])   # マシンA（部品α、部品β、部品γ）
        temp.append([ 7,  5,  3])   # マシンB（部品α、部品β、部品γ）
        temp.append([ 5,  4,  2])   # マシンC（部品α、部品β、部品γ）
        st.session_state.cap_params_list = temp

        # パラメータの初期設定（稼働率）
        st.session_state.operating_rate = 75

        # # パラメータの初期設定（未生産分のペナルティ）
        # st.session_state.incomplete_loss = 100

        # # パラメータの初期設定（作りすぎのペナルティ）
        # st.session_state.complete_loss = 20


    # stのタイトル表示
    st.title("遺伝的アルゴリズム\n（製造機器の稼働におけるCO2排出量の最適化問題)")

    # ファイルのアップローダー
    uploaded_file = st.sidebar.file_uploader("データのアップロード", type='csv') 

    # サイドメニューの設定
    activities = ["製造指示確認", "ＣＯ２排出量", "部品製造能力", "最適化の実行", "About"]
    choice = st.sidebar.selectbox("Select Activity", activities)

    if choice == '製造指示確認':
        # アップロードの有無を確認
        if uploaded_file is not None:

            # 一度、read_csvをするとインスタンスが消えるので、コピーしておく
            ufile = copy.deepcopy(uploaded_file)

            try:
                # 文字コードの判定（utf-8 bomで試し読み）
                pd.read_csv(ufile, encoding="utf_8_sig")
                enc = "utf_8_sig"
            except:
                enc = "shift-jis"

            finally:
                # データフレームの読み込み（一番左端の列をインデックスに設定）
                df = pd.read_csv(uploaded_file, encoding=enc, index_col=0) 

                # ary_cnt = ["10", "50", "100", ]
                # cnt = st.sidebar.selectbox("Select Max mm", ary_cnt)
                cnt = st.sidebar.slider('表示する件数', 1, len(df), 10)

                # テーブルの表示
                display_table('製造指示データ', df.head(int(cnt)))

                # データフレームをセッションステートに退避
                st.session_state.df_norma = copy.deepcopy(df)
        else:
            st.subheader('製造指示データをアップロードしてください')

    if choice == 'ＣＯ２排出量':

        # カラムの作成
        col1, col2, col3 = st.columns(3)

        # セッションステートから値をコピー（セッションステートをそのまま使うと遅いため）
        params_list = copy.deepcopy(st.session_state.co2_params_list)   # DeepCopyしないとダメ

        captions = ['製造時のCO2排出量(/h)', '交換時のCO2排出量(/h)', '整備時のCO2排出量(/h)', '遊休時のCO2排出量(/h)', ]

        with col1:

            st.text('マシンＡの性能')

            for idx, caption in enumerate(captions):
                params_list[0][idx] = st.number_input(caption, value=params_list[0][idx], key=params_list[0][idx] * 0)

        with col2:

            st.text('マシンＢの性能')

            for idx, caption in enumerate(captions):
                params_list[1][idx] = st.number_input(caption, value=params_list[1][idx], key=params_list[0][idx] * 1)

        with col3:

            st.text('マシンＣの性能')

            for idx, caption in enumerate(captions):
                params_list[2][idx] = st.number_input(caption, value=params_list[2][idx], key=params_list[0][idx] * 2)


        # 保存ボタンの作成（これは、この位置 = params_listの定義より後ろにないとNG）
        st.sidebar.text('パラメータの保存')
        if st.sidebar.button('保存の実行'):
            st.session_state.co2_params_list = copy.deepcopy(params_list)   # DeepCopyで値を戻す


    if choice == '部品製造能力':

        # カラムの作成
        col1, col2, col3 = st.columns(3)

        # セッションステートから値をコピー（セッションステートをそのまま使うと遅いため）
        params_list = copy.deepcopy(st.session_state.cap_params_list)   # DeepCopyしないとダメ
        op_rate = st.session_state.operating_rate

        captions = ['部品αの製造能力(/h)', '部品βの製造能力(/h)', '部品γの製造能力(/h)', ]

        with col1:

            st.text('マシンＡの性能')

            for idx, caption in enumerate(captions):
                params_list[0][idx] = st.number_input(caption, value=params_list[0][idx], key=params_list[0][idx] * 0)

        with col2:

            st.text('マシンＢの性能')

            for idx, caption in enumerate(captions):
                val = params_list[1][idx]
                params_list[1][idx] = st.number_input(caption, value=params_list[1][idx], key=params_list[0][idx] * 1)

        with col3:

            st.text('マシンＣの性能')

            for idx, caption in enumerate(captions):
                params_list[2][idx] = st.number_input(caption, value=params_list[2][idx], key=params_list[0][idx] * 2)

        op_rate = st.number_input('全体の稼働率(%)', value=op_rate)

        # 保存ボタンの作成（これは、この位置 = params_listの定義より後ろにないとNG）
        st.sidebar.text('パラメータの保存')
        if st.sidebar.button('保存の実行'):
            st.session_state.cap_params_list = copy.deepcopy(params_list)   # DeepCopyで値を戻す
            st.session_state.operating_rate =  copy.deepcopy(op_rate)


    if choice == '最適化の実行':

        # # アップロードの有無を確認
        # if uploaded_file is not None:
        # セッションステートにデータフレームがあるかを確認
        if 'df_norma' in st.session_state or 'df_norma' not in st.session_state:

            # 表示する世代
            max_individual = st.sidebar.slider('世代の個体数', value=5, min_value=5, max_value=100, step=1)
            # choice_graph = st.sidebar.selectbox("評価値の遷移グラフ", ['表示しない','表示する'])

            # データフレームの読み込み（一番左端の列をインデックスに設定） ※デバック用
            df_norma = pd.read_csv('製造指示.csv', encoding="utf_8_sig", index_col=0) 

            # データフレームをセッションステートに退避  ※デバック用
            st.session_state.df_norma = copy.deepcopy(df_norma)

            # テーブルの表示
            display_table('製造指示データ', df_norma)

            st.header('稼働状況データ')
            st.text('遊休=0, 製造=1(部品α),2(部品β),3(部品γ), 交換=9, 整備=-1')            

            # 全世代の個体を格納するリストを初期化
            df_shift_list = []

            if st.sidebar.button('第0世代を生成する'):

                for i in range(max_individual):

                    # 第0世代の生成（引数：稼働率）
                    df_shift_0th = generate_0th_generation(st.session_state.operating_rate)
                    display_individual('第0世代(個体:' + str(i) + '番)', df_shift_0th, [0, 0, 0])
                    df_shift = copy.deepcopy(df_shift_0th)
                    df_shift_list.append(df_shift)    # リストに格納

                # 世代の全個体リストをセッションステートに保存
                st.session_state.df_shift_list = df_shift_list

            # ペナルティの重み設定
            incomplete_loss = st.sidebar.number_input('生産不足のペナルティ（重み）', value=200)
            complete_loss = st.sidebar.number_input('生産過多のペナルティ（重み）', value=20)
            co2_loss = st.sidebar.number_input('ＣＯ２排出量のペナルティ（重み）', value=50)
            max_generation = st.sidebar.slider('生成する世代数(n)', value=1, min_value=1, max_value=100, step=1)

            if st.sidebar.button(f'～第{max_generation}世代までを生成する'):
                
                # 第n-1世代が存在する場合...
                if 'df_shift_list' in st.session_state:

                    # セッションステートから世代の全個体リストを復元
                    df_shift_list = st.session_state.df_shift_list

                    # リストから個体を1つずつ取り出し
                    for idx, df_shift in enumerate(df_shift_list):

                        temp_shift_list = []     # 交換(9)を挿入した行を3行まとめるためのリスト

                        # 個体から1行ずつ取り出し（マシンＡ, Ｂ, Ｃ）
                        for index, row in df_shift.iterrows():
                            temp_shift = add_unit_switch(row)     # 部品の交換をチェックして、2hの交換(9)を挿入する
                            temp_shift_list.append(temp_shift)

                        # 個体評価用のデータフレームを作成
                        df_shift_evaluation = pd.DataFrame(temp_shift_list,  index=['マシンＡ', 'マシンＢ', 'マシンＣ'])

                        # 個体を評価する
                        df_norma = st.session_state.df_norma                # 製造指示（ノルマ）を読み込み
                        cap_params_list = st.session_state.cap_params_list  # 部品製造能力を読み込み
                        co2_params_list = st.session_state.co2_params_list  # ＣＯ２排出量を読み込み

                        # 生産ノルマを守れているかの評価 ＆ ＣＯ２排出量を評価
                        loss_list = [incomplete_loss, complete_loss, co2_loss]    # ペナルティの重みをリスト化する
                        score_list = evaluation_individual(df_shift_evaluation, df_norma, cap_params_list, co2_params_list, loss_list)

                        incomplete_score = score_list[0]    # 生産不足のペナルティスコア
                        complete_score = score_list[1]      # 生産過多のペナルティスコア
                        co2_score = score_list[2]           # CO2排出量のペナルティスコア
                        total_score = score_list[0] + score_list[1] + score_list[2]     # 合計スコア

                        print('score_list')
                        print(score_list)

                        # 第n世代の表示
                        display_individual('第n世代(個体:' + str(idx) + '番)', df_shift, score_list)
                        display_individual('第n世代(個体:' + str(idx) + '番)', df_shift_evaluation, score_list)

                        # いったんここまで。次は評価の高い個体を残すアルゴリズムを選出。ベストも。


                # 世代の全個体リストをセッションステートに保存
                st.session_state.df_shift_list = df_shift_list    
                                
        else:
            st.subheader('製造指示データをアップロードしてください')


    if choice == 'About':

        image = Image.open('logo_nail.png')
        st.image(image)

        st.markdown("Built by [Nail Team]")
        st.text("Version 0.2")
        st.markdown("For More Information check out   (https://nai-lab.com/)")
        

if __name__ == "__main__":
    main()


