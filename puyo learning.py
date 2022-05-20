#――DQN学習プログラムとぷよぷよプログラムを繋ぐメインプログラム――

# ライブラリをインポート
import DQN # 自作学習プログラム
import puyo # ぷよぷよプログラム（学習環境）

import os
import random
import tkinter as tk # GUIライブラリ
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg # tkinterでもmatplotlibを使えるようにする奴
from functools import partial # 部分適用の奴（一部引数を事前に決めておく奴）


# ――定数・パラメータ設定――
# ぷよぷよ関連
puyo.PUYO_COLOR_NUM = 3 # 学習を簡単にするため、ぷよの色数を減らす
WAIT_AI = 60 # AIの操作間隔

# DQN学習関連
DQN.ETA = 0.001 # 学習率
DQN.GAMMA = 0.99 # 報酬の割引率(将来の報酬の考慮度合い)

DQN.EPSILON_MAX = 1.0 # ε（ランダム行動の確率）の初期値
DQN.EPSILON_REDUCE_STEPS = 1/300000 # εの1stepあたりの減少度
DQN.EPSILON_MIN = 0.01 # εの最小値

DQN.TARGET_UPDATE_INTERVAL = 30000 # target_networkを更新する頻度（あまり頻繁でなくて良い）

DQN.INITIAL_REPLAY_SIZE = 500 # 学習前に確保しておく経験の量
DQN.BATCH_CAPACITY = 200000 # ReplayMemoryの容量
DQN.BATCH_SIZE = 16 # 1回のミニバッチ学習で用いるデータ量

# システム関連
EPISODE_NUM = 200 # １エピソード辺りの最大ステップ数
GRAPH_INTERVAL = 10 # グラフの描画間隔(episodes/1回)
GRAPH_DATASTOCK = 10000 # グラフ用に保存しておくデータの最大数(episode)。これを超えると、古いものからデータを削除していく。



# 環境やエージェントの全体管理クラス
class MainControl:
    def __init__(self):
        #ウィンドウ作成
        self.app = tk.Tk()
        self.app.geometry(str(puyo.WINDOW_WIDTH) + "x" + str(puyo.WINDOW_HEIGHT))
        self.app.title("ぷよぷよ")
        
        # 各ボタン（開始ボタン以外）を設置（開始ボタンはpuyoのEventHandllerが勝手に設置する）
        self.button2 = tk.Button(self.app, text='ＡＩプレイ', command=self.AI_play_ready)
        self.button2.place(x=puyo.WINDOW_PADDING + puyo.BLOCK_SIZE * (puyo.FIELD_WIDTH + 2) + puyo.WINDOW_PADDING, y=puyo.WINDOW_PADDING + 45)
        self.button3 = tk.Button(self.app, text='学習開始', command=self.train)
        self.button3.place(x=puyo.WINDOW_PADDING + puyo.BLOCK_SIZE * (puyo.FIELD_WIDTH + 2) + puyo.WINDOW_PADDING, y=puyo.WINDOW_PADDING + 85)

        # エージェントと環境の用意
        n_state = (puyo.FIELD_HEIGHT + 1) * puyo.FIELD_WIDTH + 2 # 状態を構成する要素数（フィールドのマスの数＋操作ぷよの２色）
        n_action = puyo.FIELD_WIDTH * 4 - 2 # 行動の種類数（操作ぷよの横座標×操作ぷよの回転方向－両端の絶対に不可能な２パターン）

        self.agent = Agent(self.app, n_state, n_action, n_hidden=100)

        self.environment = PuyoEnvironment(self, self.app)

        self.app.mainloop()

    def button_destroy(self): # 開始、ＡＩプレイ、学習開始のボタンを全て消去する
        self.environment.NomalEventHandller.button.destroy()
        self.button2.destroy()
        self.button3.destroy()

#ＡＩプレイモード――――――――
    def AI_play_ready(self): # AIプレイモードの準備
        self.button_destroy()

        # プレイするＡＩデータファイルの選択
        file_path = tk.filedialog.askopenfilename(
            initialdir="./PTH_files/", # 初期表示フォルダ
            filetypes=[("PTH", "*.pth")], # 選択対象
            title = "AIデータを選択" # タイトル
        )
        self.agent.load(file_path) # 学習データの読み込み

        # 詳細情報の欄の作成
        self.cnt_gameover = 0 # ゲームオーバーの回数
        self.max_ren = 0 # 最大連鎖数
        self.cnt_ren = 0 # 連鎖を開始した回数
        self.sum_ren = 0 # 連鎖数の総和
        self.cnt_4ren = 0 # 4連鎖以上を行った回数
        self.cnt_miss = 0 # 判断ミスをした回数（設置不可能な場所へ操作ぷよを運ぼうとした回数）
        self.textstr = [tk.StringVar() for i in range(5)]
        self.text = [tk.Label(self.app, font=("", 10), textvariable=self.textstr[i]) for i in range(5)]
        for i in range(5):
            self.text[i].place(x=puyo.BLOCK_SIZE * (puyo.FIELD_WIDTH + 2) + puyo.WINDOW_PADDING + 5, y=puyo.WINDOW_PADDING + 20 * i)
        self.textstr[0].set("終了回数：0")
        self.textstr[1].set("最大連鎖数：0")
        self.textstr[2].set("平均連鎖数：0")
        self.textstr[3].set("4連鎖以上：0")
        self.textstr[4].set("判断ミス：0")

        # ゲームのスタート
        self.put_flag = False
        self.environment.game.set_func(self.AI_play)
        self.environment.game.start(end_func=self.AI_play_restart)

    def AI_play(self): # ＡＩプレイモード用の関数
        # 詳細情報の更新（連鎖数関連）
        if self.put_flag and self.environment.game.get_ren() > 0 :
            ren = self.environment.game.get_ren()
            self.cnt_ren += 1
            self.sum_ren += ren
            if ren > self.max_ren : self.max_ren = ren
            if ren >= 4 : self.cnt_4ren += 1
        else : self.put_flag = True

        # 詳細情報の描画更新
        self.textstr[1].set("最大連鎖数：" + str(self.max_ren))
        if self.cnt_ren : self.textstr[2].set("平均連鎖数：" + str(self.sum_ren / self.cnt_ren * 1000 // 10 / 100))
        if self.environment.game.get_ren() >= 4 : self.textstr[3].set("4連鎖以上：" + str(self.cnt_4ren))

        state = self.environment.get_state() # 状態の取得
        action = self.agent.select_action(state) # 行動の選択
        self.environment.AIplay_update(action) # 行動を実行

    def AI_play_restart(self): # ＡＩプレイモードでゲームオーバーになったときの処理（ゲームをリセットするだけ）
        self.put_flag = False
        self.cnt_gameover += 1

        # 詳細情報の描画更新
        self.textstr[0].set("終了回数：" + str(self.cnt_gameover))
        self.textstr[1].set("最大連鎖数：" + str(self.max_ren))
        if self.cnt_ren : self.textstr[2].set("平均連鎖数：" + str(self.sum_ren / self.cnt_ren * 1000 // 10 / 100))
        if self.environment.game.get_ren() >= 4 : self.textstr[3].set("4連鎖以上：" + str(self.cnt_4ren))

        # ゲーム再開
        self.environment.game.set_func(self.AI_play)
        self.environment.game.start(end_func=self.AI_play_restart)

#学習モード――――――――
    def train(self): # 学習の準備段階
        self.AI_id = random.randint(10000, 99999) # 適当な乱数を決めておく
        self.button_destroy()
        self.environment.train_ready()
        self.agent.init_plot(update_interval=GRAPH_INTERVAL, datastock=GRAPH_DATASTOCK) # グラフの描画用意

        # 学習終了（もしくは学習結果保存）、学習速度変更の二つのボタンを追加（一時停止ボタンはいらないかも）
        self.button2 = tk.Button(self.app, text='結果保存', command=self.save_train_data)
        self.button2.place(x=puyo.WINDOW_PADDING + puyo.BLOCK_SIZE * (puyo.FIELD_WIDTH + 2) + puyo.WINDOW_PADDING, y=puyo.WINDOW_PADDING + 5)
        self.button3 = tk.Button(self.app, text='速度変更', command=self.change_train_speed)
        self.button3.place(x=puyo.WINDOW_PADDING + puyo.BLOCK_SIZE * (puyo.FIELD_WIDTH + 2) + puyo.WINDOW_PADDING, y=puyo.WINDOW_PADDING + 45)

        # 初期化して学習を開始
        self.step = 0
        self.before_do = True
        self.environment.game.set_func(self.train_restart)
        self.environment.game.start(end_func=self.train_episode2)

    def train_episode1(self): # 実際に学習する（その１、行動選択）
        self.state = self.state_next # 状態を更新
        self.action = self.agent.select_action(self.state, True) # 行動選択
        self.environment.update(self.action, self.before_do) # 行動実行（→train_episode2へ）

    def train_episode2(self, do=True): # 実際に学習する（その２、行動実行後）（doがFalseのとき、実行不可能な行動を指定した）
        # 行動の結果を取得・計算
        self.step += 1 # ステップ数増加（一定ステップでエピソード強制終了）
        self.before_do = do # 適切に行動を選択できたかを記録
        self.state_next = self.environment.get_state() # 行動を行った後の状態を取得
        #self.reward = np.sign(self.environment.calc_reward(do)) # 報酬を決定
        self.reward = self.environment.calc_reward(do) # 報酬を決定

        # エピソードが終了したかを判定
        if self.step >= EPISODE_NUM or self.environment.game.field.is_gameover() : # ゲームオーバーか一定ステップ経過のどちらかで終了
            self.done = True
        else :
            self.done = False

        # 元の状態、行動、報酬、次の状態、終了判定より、エージェントが学習を行う
        self.agent.train(self.state, self.action, self.reward, self.state_next, self.done)

        #エピソードが終了したとき、ゲームをリセットする
        if self.done :
            self.step = 0
            self.environment.reset(end_func=self.train_episode2)

        else :
            # 終了していないとき、train_episode1を呼び出す
            self.environment.root.after(0, self.train_episode1)

    def train_restart(self): # 学習でエピソードが終了した際、リスタートした後の細々とした処理
        self.state_next = self.environment.get_state() # 初期状態を取得
        self.environment.root.after(0, self.train_episode1)

    def save_train_data(self): # 学習結果をファイルに保存する
        # 保存するときのファイル名を決める。自動生成。（「ID(=乱数) + 学習エピソード数 ( + 被り回避番号)」で決められる）
        base_name = "./PTH_files/model" + str(self.AI_id) + "_ep" + str(self.agent.episodes)
        add_name = ""
        number = 1
        if os.path.isfile(base_name + ".pth") :
            # ファイル名が被っているとき
            while os.path.isfile(base_name + "_" + str(number) + ".pth"):
                number += 1
            add_name = "_" + str(number)
        self.agent.save(base_name + add_name + ".pth")

    def change_train_speed(self): # 学習中の速度を変更する
        if puyo.PUT_WAIT :
            #超高速（ウェイトなし）
            puyo.DRAW_SYSTEM = False
            puyo.PUT_WAIT = 0
            puyo.REN_WAIT = 0
            self.environment.game.canvas.text.set("描画一時停止中...") # 連鎖の所にテキスト表示
            self.environment.game.field.set_ren(-1) # 次の描画更新でテキストが更新されるように-1をセットしておく

        else :
            #ゆっくり（ウェイトあり。経過観察用）
            puyo.DRAW_SYSTEM = True
            puyo.PUT_WAIT = 300
            puyo.REN_WAIT = 600



# 学習環境クラス（DQNのEnvironmentクラス、puyopuyoのEventHandllerクラスの役割を持つ）
class PuyoEnvironment:
    def __init__(self, main, root):
        self.main = main # 全体制御クラスの確認

        self.root = root # ウィンドウ確認

        self.game = puyo.PuyoGame(root) # ゲーム作成

        self.NomalEventHandller = EventHandller(root, self.game, main.button_destroy) # 通常プレイ用のイベントハンドラ作成

        # 色の名前を数値に変換する連想配列
        self.color_num = {}
        for color in range(puyo.PUYO_COLOR_NUM + 1) :
            self.color_num[puyo.PUYO_COLOR[color]] = color

    def get_block(self, x, y): # フィールドの指定の座標のマスの色を数値で取得する（0:空き、1~:対応する色のぷよ有り）
        return self.color_num[self.game.field.get_block(x, y).get_color()]

    def get_state(self): # 学習用の形式で状態を返す
        state = []
        # 各フィールドの状態を追加
        for i in range((self.game.field.get_height() + 1) * self.game.field.get_width()):
            state.append(self.color_num[self.game.field.get_blocks()[i].get_color()])

        # 操作ぷよの色を追加
        color1, color2 = self.game.puyo.get_color(0)
        state.append(color1)
        state.append(color2)

        return state # フィールドの状態＋操作ぷよの色

    def calc_reward(self, do): # 報酬を計算する　目標：ゲームオーバーせずに、大きな連鎖ができるＡＩ（同時消し・連結数は考慮せず）
        if not do : return -2 # 実行不可能な行動を選択した＝最も大きな罰
        elif self.game.field.is_gameover() : return -1.5 # ゲームオーバーになった＝大きな罰
        else :
            ren = self.game.field.get_ren()
            if ren == 0 : return 0
            else : return ren / 2 - 1 # それ以外＝連鎖数に応じて報酬を与える

    def train_ready(self): # 学習に向けて準備を行う
        #学習用にぷよぷよの設定変更
        puyo.DRAW_CONTROL = False
        puyo.DRAW_DETAIL = False
        puyo.DRAW_SYSTEM = True
        puyo.PUT_WAIT = 300
        puyo.FALL_WAIT = 0
        puyo.REN_WAIT = 600

    def update(self, action, before_do=True): # 行動を元に処理の更新を行う（戻り値：次の状態, 報酬, 終了判定）
        # 操作ぷよを指定の状態にする
        x = (action + 1) // 4
        if action >= 20 : angle = (action + 2) % 4
        elif action >= 3 : angle = (action + 1) % 4
        else : angle = action % 4

        # 操作では指定の場所に運べないとき、罰を与えて行動選択のし直し。
        if before_do : self.is_can_move_ready() # 行動を起こしたらリセットをかける。
        if not self.is_can_move(x, angle) :
            self.main.train_episode2(do=False)

        else :
            # 操作ぷよを指定位置の上方に移動
            self.game.puyo.set_coord(x, -1)
            self.game.puyo.set_angle(angle)

            # 地面に着くまで落下させる
            while self.game.field.is_can_move(self.game.puyo, move_direction=puyo.MOVE_DOWN) :
                self.game.puyo.set_coord(x, self.game.puyo.get_coord()[1] + 1)

            # 設置後の処理へ移行
            self.game.field.put_puyo(self.game.puyo)
            self.game.puyo.set_exist(False)
            
            # puyoの接地後の処理を呼び出して、その後maincontrolのtrain_episode2関数を実行する
            self.game.set_func(partial(self.main.train_episode2, do=True))
            self.game.fall_loop()

    def reset(self, end_func): # 処理のリセットを行う（戻り値：最初の状態）
        self.game.set_func(self.main.train_restart)
        self.game.start(end_func=end_func)

    def is_can_move_ready(self): # is_can_moveで判定を行うための準備。フィールドの状態から、可能な操作などを分析する（フィールド幅６列用）
        self.blank_row = [False, False, False, False, False, False] # 列の最上段が空いているか
        self.move_row = [False, False, False, False, False, False] # まわしなどを用いて、操作によって指定の列へぷよを運べるか

        #列の最上段が空いているかの判定
        for x in range(6) :
            if self.get_block(x, 0) :
                self.blank_row[x] = True
                #最上段が埋まっている＝そこより奥には運べない
                if x < 2 :
                    self.move_row[:x + 1] = [True for i in range(x + 1)]
                else :
                    self.move_row[x:] = [True for i in range(6 - x)]

        #操作で指定の位置へ運べるかの判定
        if self.get_block(1, 1) != 0 and self.get_block(3, 1) + self.get_block(2, 2) == 0 :
            self.move_row[:2] = [True, True] # ２列目の上部が埋まり、まわせる状態でもない
        elif self.get_block(0, 1) != 0 and self.get_block(1, 2) == 0 :
            self.move_row[0] = True # １列目の上部が埋まり、まわせる状態でもない

        if self.get_block(3, 1) != 0 and self.get_block(1, 1) + self.get_block(2, 2) == 0 :
            self.move_row[3:] = [True, True, True] # ４列目の上部が埋まり、まわせる状態でもない
        elif self.get_block(4, 1) != 0 and self.get_block(3, 2) == 0 :
            self.move_row[4:] = [True, True] # ５列目の上部が埋まり、まわせる状態でもない
        elif self.get_block(5, 1) != 0 and self.get_block(3, 1) + self.get_block(4, 2) == 0 :
            self.move_row[5] = True # ６列目の上部が埋まり、まわせる状態でもない

    def is_can_move(self, x, angle): # 指定の場所まで、人間の操作でも運んでいけるかの判定（事前にis_can_move_readyを呼び出すこと）（フィールド幅６列用）
        #１列目
        if x==0 :
            if self.move_row[0] : return False # 指定の列へ到達不可能
        #２列目
        elif x==1 :
            if self.move_row[1] : return False # 指定の列へ到達不可能
            if angle==puyo.MOVE_LEFT and self.move_row[0] : return False # 左の列が空いておらず、回転方向が左の場合不可能
        #３列目（回転する場合）
        elif x==2 and angle!=puyo.MOVE_UP :
            if self.blank_row[1] and self.blank_row[3] : return False # 両端の最上段が埋まっており、回転不可能
            if angle==puyo.MOVE_LEFT and self.move_row[1] : return False # 左の列が空いておらず、回転方向が左の場合不可能
            if angle==puyo.MOVE_RIGHT and self.move_row[3] : return False # 右の列が空いておらず、回転方向が右の場合不可能
        #４列目
        elif x==3 :
            if self.move_row[3] : return False # 指定の列へ到達不可能
            if angle==puyo.MOVE_RIGHT and self.move_row[4] : return False # 右の列が空いておらず、回転方向が右の場合不可能
        #５列目
        elif x==4 :
            if self.move_row[4] : return False # 指定の列へ到達不可能
            if angle==puyo.MOVE_RIGHT and self.move_row[5] : return False # 右の列が空いておらず、回転方向が右の場合不可能
        #６列目
        elif x==5 and self.move_row[5] :
            return False # 指定の列へ到達不可能

        return True

#ＡＩプレイモード用――――――――
    def AIplay_update(self, action): # ＡＩプレイモード用。選択した行動によって、処理と描画を更新する。
        # 指定している操作ぷよの状態を計算する
        x = (action + 1) // 4
        if action >= 20 : angle = (action + 2) % 4
        elif action >= 3 : angle = (action + 1) % 4
        else : angle = action % 4

        print(str(x) + ", " + str(angle))

        self.is_can_move_ready()
        if not self.is_can_move(x, angle) : # 操作によって指定の場所まで運ぶのは不可能だったとき
            print("impossible selection!")
            self.main.cnt_miss += 1
            self.main.textstr[4].set("判断ミス：" + str(self.main.cnt_miss))
            self.AI_fall(1000) # ゆっくりと真下に落とす

        else :
            # 操作によって、指定の場所まで運ぶ
            self.game.set_func(partial(self.control, x=x, angle=angle)) # 操作したら次の操作をするために戻ってくる
            self.control(x, angle)

    def control(self, x, angle): # ＡＩプレイモード用。操作ぷよを指定の場所まで、キー入力にて運ぶ（人力でも再現可能なことを示すため）
        px, py =  self.game.puyo.get_coord()
        pangle = self.game.puyo.get_angle()

        # 左右移動・回転完了したら下入力へ移行する
        if px == x and pangle == angle :
            self.AI_fall(WAIT_AI)
            return

        # 左右移動・回転操作を行う（判定を一から毎回行うので非効率的だが、負荷も気にならない程度なのでそのまま）
        if x == 2 and angle != puyo.MOVE_UP :
            # 開始場所から左右移動しないとき（回転方向について、右回転か左回転かを考える）
            block11 = (self.get_block(1, 1) != 0) # フィールド(1, 1)が埋まっているか
            block30 = (self.get_block(3, 0) != 0) # フィールド(3, 0)が埋まっているか
            block31 = (self.get_block(3, 1) != 0) # フィールド(3, 1)が埋まっているか
            rightroll_tower = block31 and block11 and not block30 # 両隣が積み上がっており、かつ少なくとも右回転のみで任意の回転状態にできる状況かどうか
            left_natural = (angle == puyo.MOVE_LEFT and (rightroll_tower or not block11)) # 左回転の方が自然である場合かどうか

            if (rightroll_tower or not block31) and not left_natural:
                self.root.after(WAIT_AI, lambda : self.game.roll_puyo(direction=puyo.ROLL_RIGHT)) # 右回転を選択
            else :
                self.root.after(WAIT_AI, lambda : self.game.roll_puyo(direction=puyo.ROLL_LEFT)) # 左回転を選択
        
        elif px != x :
            # 開始場所からまず左右移動を行うとき（まわしが必要なときは回転も使用する）
            direction = 1 - 2 * (x < 2) # 移動方向の確認
            if (self.get_block(px + direction, 1) != 0 and py == 1) or pangle != puyo.MOVE_UP :
                # まわしが必要な状況である
                if py == 1 :
                    self.root.after(WAIT_AI, lambda : self.game.roll_puyo(direction=4.5 + 0.5 * direction)) # 位置を一段上げるために回転させる
                else :
                    self.root.after(WAIT_AI, lambda : self.game.roll_puyo(direction=4.5 - 0.5 * direction)) # 変えた回転方向を元に戻す
            else :
                # 左右移動で行ける
                self.root.after(WAIT_AI, lambda : self.game.move_puyo(direction = 2 - direction)) # 左右移動

        else :
            # 移動を終えた後の回転（移動なしの場合よりも考えることは少ない）
            if px == 0 :
                self.root.after(WAIT_AI, lambda : self.game.roll_puyo(direction=puyo.ROLL_RIGHT)) # 右回転を選択（左端）
            elif px == 5 :
                self.root.after(WAIT_AI, lambda : self.game.roll_puyo(direction=puyo.ROLL_LEFT)) # 左回転を選択（右端）
            else :
                direction = 1 - 2 * (x < 2)
                if angle == 2 - direction and self.get_block(x + direction, py) == 0 :
                    self.root.after(WAIT_AI, lambda : self.game.roll_puyo(direction = 4.5 - 0.5 * direction)) # 回転数の少ない回転方向を選択
                else :
                    self.root.after(WAIT_AI, lambda : self.game.roll_puyo(direction = 4.5 + 0.5 * direction)) # 基本は×マークの方向で回転

    def AI_fall(self, timer): # ＡＩプレイモード用。操作ぷよを接地するまで落下させる。
        if self.game.field.is_can_move(self.game.puyo, move_direction=puyo.MOVE_DOWN) :
            # 下に落とせるとき、落下処理をして再び自分を呼び出す
            self.game.set_func(partial(self.AI_fall, timer=timer))
            self.root.after(timer, lambda : self.game.move_puyo(puyo.MOVE_DOWN))

        else :
            # 地面に着いたとき、接地処理をしてmaincontrolのAIplayを呼び出す
            self.game.set_func(self.main.AI_play)
            self.root.after(timer, lambda : self.game.move_puyo(puyo.MOVE_DOWN))



# ぷよぷよの通常プレイ用のイベントハンドラ（GUIの変更に合わせて、若干の修正を加えた）
class EventHandller( puyo.EventHandller ):
    def __init__(self, root, game, func):
        self.buttondestroy = func # 開始以外のボタンの消去命令を受け取り
        super().__init__(root, game)

    def start_event(self):
        self.buttondestroy() # ゲーム開始時に、全てのボタンの消去命令を実行
        super().start_event()


# グラフをtkinterに対応させるために、Agentを多少変更する
class Agent( DQN.Agent ):
    def __init__(self, root, n_state, n_action, n_hidden=32, eta=DQN.ETA, gamma=DQN.GAMMA, epsilon_max=DQN.EPSILON_MAX, epsilon_reduce_steps=DQN.EPSILON_REDUCE_STEPS, epsilon_min=DQN.EPSILON_MIN, target_update_interval=DQN.TARGET_UPDATE_INTERVAL, initial_replay_size=DQN.INITIAL_REPLAY_SIZE, batch_capacity=DQN.BATCH_CAPACITY, batch_size=DQN.BATCH_SIZE):
        self.root = root
        super().__init__(n_state, n_action, n_hidden=n_hidden, eta=eta, gamma=gamma, epsilon_max=epsilon_max, epsilon_reduce_steps=epsilon_reduce_steps, epsilon_min=epsilon_min, target_update_interval=target_update_interval, initial_replay_size=initial_replay_size, batch_capacity=batch_capacity, batch_size=batch_size)

    def build_graph(self):
        return Graph(self.root) # ここで作成したGraphクラスを指定する


# グラフをtkinterに対応させるために変更を加える
class Graph( DQN.Graph ):
    def __init__(self, root):
        super().__init__()
        self.root = root # ウィンドウの確認

    def ready(self, update_interval, datastock):
        super().ready(update_interval, datastock)
        # ウィンドウのサイズ変更
        self.x = puyo.BLOCK_SIZE * (puyo.FIELD_WIDTH + 2) + puyo.WINDOW_PADDING * 2 + 80 # キャンバスを設置するx座標を計算
        self.root.geometry(str(max([self.x + 800 + puyo.WINDOW_PADDING, puyo.WINDOW_WIDTH]))+"x"+str(max([puyo.WINDOW_PADDING * 2 + 800, puyo.WINDOW_HEIGHT]))) # ウィンドウサイズ変更
        self.root.resizable(width=False, height=False) # ウィンドウのサイズ固定

        # キャンバス生成
        self.canvas = FigureCanvasTkAgg(self.fig, self.root) # キャンバス生成
        self.canvas.draw() # キャンバス描画
        self.canvas.get_tk_widget().place(x=self.x, y=puyo.WINDOW_PADDING) # キャンバス配置

    def update(self, episodes, step, total_reward, avg_loss, avg_MaxQ): # 数エピソード毎に更新されるタイプのグラフ用の更新関数（エピソードの終了時に毎度呼び出される）
        super().update(episodes, step, total_reward, avg_loss, avg_MaxQ)
        episodes = self.cnt_episodes[len(self.cnt_episodes) - 1]
        if episodes < 2 or episodes % self.update_interval != 0 : return # 1エピソード分の情報しかないときは描画更新しない・一定のエピソード数毎に描画更新を行う

        self.dataset(episodes) # データ更新
        self.canvas.draw() # キャンバス描画
        self.canvas.get_tk_widget().place(x=self.x, y=puyo.WINDOW_PADDING) # キャンバス配置


if __name__=='__main__' :
    MainControl()
