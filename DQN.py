"""
    DQN学習プログラム
"""
# ライブラリをインポートする
from typing import List
import torch # 機械学習ライブラリ
import torch.nn as nn

import copy
import random
import numpy as np # 計算ライブラリ
import matplotlib.pyplot as plt # グラフ描画ライブラリ


# ハイパーパラメータの設定
ETA = 0.001 # 学習率
GAMMA = 0.9 # 報酬の割引率(将来の報酬の考慮度合い)

EPSILON_MAX = 1.0 # ε（ランダム行動の確率）の初期値
EPSILON_REDUCE_STEPS = 1/10000 # εの1stepあたりの減少度
EPSILON_MIN = 0.01 # εの最小値

TARGET_UPDATE_INTERVAL = 10000 # target_networkを更新する頻度（あまり頻繁でなくて良い）

INITIAL_REPLAY_SIZE = 100 # 学習前に確保しておく経験の量
BATCH_CAPACITY = 20000 # ReplayMemoryの容量
BATCH_SIZE = 8 # 1回のミニバッチ学習で用いるデータ量



# 乱数ジェネレータの定義
rng = np.random.default_rng()
# デバイスの設定（モデルとtensorをto(device)で切り替える。ndarrayはcpuにしてから変換する）
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


#ndarray型をTensor型に変換する関数
def toTensor(ndarray: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(ndarray.astype(np.float32)).clone().to(device)

#Tensor型をndarray型に変換する関数
def toNdarray(tensor: torch.Tensor) -> np.ndarray:
    if tensor.is_cuda : tensor = tensor.cpu()
    return tensor.detach().numpy()


# 学習を行うエージェントのクラス
class Agent:
    # デバッグ用フラグ
    Debug_Action = True # 各ステップの行動選択の経過をターミナルに表示
    Debug_MiniBatch = False # 各ミニバッチ学習の経過をターミナルに表示

    def __init__(self, n_state: int, n_action: int, n_hidden: int=32, eta: float=ETA, gamma: float=GAMMA, epsilon_max: float=EPSILON_MAX, epsilon_reduce_steps: float=EPSILON_REDUCE_STEPS, epsilon_min: float=EPSILON_MIN, target_update_interval: int=TARGET_UPDATE_INTERVAL, initial_replay_size: int=INITIAL_REPLAY_SIZE, batch_capacity: int=BATCH_CAPACITY, batch_size: int=BATCH_SIZE) -> None:
        # 定数の設定
        self.gamma = gamma
        self.epsilon_max = epsilon_max
        self.epsilon_reduce_steps = epsilon_reduce_steps
        self.epsilon_min = epsilon_min
        self.target_update_interval = target_update_interval
        self.initial_replay_size = initial_replay_size
        self.batch_size = batch_size

        # ニューラルネットワークの構築
        self.n_state = n_state # 状態の数
        self.n_action = n_action # 行動の数
        self.main = self.build_model(n_state, n_action, n_hidden) # メインネットワークの構築
        self.target = copy.deepcopy(self.main) # ターゲットネットワークの構築（メインのコピー）

        # 学習についての損失関数とオプティマイザーの設定
        self.criterion = nn.HuberLoss() # 損失関数にはHuberLossを使用
        self.optimizer = torch.optim.Adam(self.main.parameters(), lr=eta) # オプティマイザーにはAdamを使用

        # ReplayMemoryの作成
        self.ReplayMemory = self.build_replaymemory(batch_capacity)

        # グラフ用変数の用意
        self.graph = self.build_graph()

        # 学習の状況を示す変数
        self.steps = 0
        self.episodes = 0
        
        # ステータス計算用変数（良さげな値をいくつか記録しておく）
        self.start_episode = 0 # エピソードが始まったときのステップ数
        self.sum_total_rewards = 0 # そのエピソードでの報酬の総和
        self.sum_loss = 0 # そのエピソードでの誤差の総和
        self.sum_MaxQ = 0 # そのエピソードでのMaxQの総和

    def build_model(self, n_state: int, n_action: int, n_hidden: int): # ニューラルネットワークの構築
        return Model(n_state, n_action, n_hidden).to(device)

    def build_replaymemory(self, batch_capacity: int): # ReplayMemoryの構築
        return ReplayMemory(batch_capacity)

    def build_graph(self): # 結果描画用グラフのクラスの作成
        return Graph()

    def select_action(self, state: List[float], mode: bool=False) -> int: # 行動選択（modeがTrueのとき、ランダムな選択を考慮する）
        # Q値の計算
        state_tensor = torch.tensor(state, dtype=torch.float, device=device) # 状態stateをtorch.tensor型に変換する
        Qpred = self.main(state_tensor) # Q値を計算

        # ランダム行動確率εの計算（徐々に最適行動を多くとるようになる）
        epsilon = self.epsilon_min + (self.epsilon_max - self.epsilon_min) / (1 + self.steps * self.epsilon_reduce_steps)

        # 行動の決定
        if mode and rng.random() < epsilon : action = rng.integers(self.n_action) # ランダムに行動を選択する(modeがTrueのとき)
        else : action = torch.argmax(Qpred).item() # 価値の高い行動を選択する

        # 経過を表示
        if self.Debug_Action :
            print(f'episode:{self.episodes} | step:{self.steps} | 状態:{state} | 選択:{action} | ε:{np.floor(epsilon*100000)/1000} | 出力:{Qpred}')

        return action

    def train(self, state: List[float], action: int, reward: float, state_next: List[float], done: bool=False) -> None: # 学習を行う
        if state_next==None : done = True # 一応、終了判定doneを自動で判定できるようにしておく

        # ReplayMemoryに経験を保存する
        state_tensor = torch.tensor(state, dtype=torch.float, device=device)
        state_next_tensor = torch.tensor(state_next, dtype=torch.float, device=device)
        self.ReplayMemory.push(state_tensor, action, reward, state_next_tensor, done)

        # グラフ用にMaxQとrewardの値を保存
        self.sum_total_rewards += reward
        self.sum_MaxQ += torch.max(self.main(state_tensor)).item()

        # ReplayMemoryより、ミニバッチ学習を行う
        if( len(self.ReplayMemory) >= self.initial_replay_size ) : # 経験が十分に集まってから、重みの調整を開始する
            if self.Debug_MiniBatch : print("ミニバッチ学習開始")
            
            samples = self.ReplayMemory.sample(self.batch_size) # ReplayMemoryより、サンプルデータを取得
            self.optimizer.zero_grad() # 勾配の値を初期化
            for sample in samples:
                # サンプルの各情報を取得
                sa_state_tensor, sa_action, sa_reward, sa_state_next_tensor, sa_done = sample

                # 状態stateに対するQ値をmain_networkで計算
                Qpred: torch.Tensor = self.main(sa_state_tensor)

                # 次に得られるQ値の最大値をtarget_networkより予測
                if sa_done : MaxQ = 0 # エピソードが終了するときは0
                else :
                    # ターゲットネットワークよりQ値を計算し最大値を取得
                    MaxQ = torch.max(self.target(sa_state_next_tensor)).item()

                # 教師信号signalの作成
                signal = Qpred.clone()
                signal[sa_action] = sa_reward + self.gamma * MaxQ

                # 損失関数により、誤差を計算する
                loss: torch.Tensor = self.criterion(Qpred, signal) / self.batch_size # 誤差を計算、平均を求める
                self.sum_loss += loss.item()

                # 勾配を計算する
                loss.backward()

                # 経過を表示
                if self.Debug_MiniBatch : print(f'    出力:{Qpred} | 選択:{sa_action} | MaxQ:{MaxQ} | 教師信号:{signal} | 誤差:{loss}')

            self.optimizer.step() # ミニバッチのすべての勾配の平均を元に、重みを更新する

            # target_networkを定期的に更新する（メインネットワークと重みを同期する）
            if self.steps % self.target_update_interval == 0 :
                self.target = copy.deepcopy(self.main)

        # ステップの増加
        self.steps += 1

        #エピソード終了時の処理
        if done :
            # グラフの更新関数の呼び出し
            if self.graph.is_ready() :
                step = self.steps - self.start_episode # 今回のエピソード終了までのstep数の計算
                self.graph.update(self.episodes + 1, step, self.sum_total_rewards, self.sum_loss / step, self.sum_MaxQ / step)

            # エピソード数の更新・計算用変数の初期化
            self.episodes += 1
            self.start_episode = self.steps
            self.sum_total_rewards = 0
            self.sum_loss = 0
            self.sum_MaxQ = 0

    def init_plot(self, update_interval, datastock): # 学習経過を示すグラフの描画用意をする
        self.graph.ready(update_interval, datastock)

    def show(self): # 学習結果を示すグラフを表示する
        if self.graph.is_ready() : self.graph.plot()

    def save(self, path): # 学習した重みの保存
        self.main.save(path)

    def load(self, path): # 学習した重みの読み込み
        self.main.load(path)
        self.target.load(path)

    def get_steps(self): # 総step数の取得
        return self.steps
    
    def get_episodes(self): # 総エピソード数の取得
        return self.episodes

    def get_n_state(self): # 状態の数の取得
        return self.n_state

    def get_n_action(self): # 行動の数の取得
        return self.n_action



# 学習での経験を保存しておくクラス
class ReplayMemory:
    def __init__(self, capacity: int):
        self.capacity = capacity # 経験の保存容量
        self.memory = [] # 経験を保存しておくリスト
        self.index = 0 # 保存するインデックス

    def push(self, state: torch.Tensor, action: int, reward: float, state_next: torch.Tensor, done: bool): # (state, action, reward, state_next, done)をReplayMemoryに保存する
        # ストックが容量以下の場合、配列に追加する場所を確保する
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # 追加処理
        self.memory[self.index] = [state, action, reward, state_next, done]
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size: int): # 指定バッチ数のサンプルを取り出す
        return random.sample(self.memory, batch_size)

    def __len__(self): # lenでメモリのストック数を得られるようにしておく
        return len(self.memory)



# ニューラルネットのモデルクラス
class Model( nn.Module ):
    def __init__(self, n_state, n_action, n_hidden=32):
        super(Model, self).__init__()

        #ニューラルネットワークの構造を決める
        self.fc = nn.Sequential(
            nn.Linear(n_state, n_hidden),
            nn.ReLU(),

            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            #nn.Sigmoid(),

            nn.Linear(n_hidden, n_action),
        )

    def forward(self, x): # 順方向の出力計算
        x = self.fc(x)
        return x

    def save(self, path): # 重みをファイルに保存する
        torch.save(self.state_dict(), path)
        

    def load(self, path): # 重みをファイルから読み込む
        self.load_state_dict(torch.load(path, map_location=device))



# グラフの管理クラス
class Graph:
    def __init__(self):
        self.ready_flag = False # 描画準備完了フラグ

        #描画用の学習状況保存リスト
        self.cnt_episodes = []
        self.steps_per_episodes = []
        self.total_rewards = []
        self.avg_loss_history = []
        self.avg_MaxQ_history = []

    def ready(self, update_interval, datastock): # 描画の準備をする
        # ※lossは、最初の経験を集める時間中は0のまま
        self.ready_flag = True
        self.update_interval = update_interval
        self.datastock = datastock

        # グラフのフォントサイズの変更
        parameters = {'axes.labelsize': 8,
                      'axes.titlesize': 10,
                      'xtick.labelsize': 5,
                      'ytick.labelsize': 5}
        plt.rcParams.update(parameters)

        # グラフの用意
        self.fig = plt.figure(figsize=(4,4))
        plt.subplots_adjust(wspace=0.5, hspace=0.5)

        # １エピソードごとのステップ数のグラフ
        self.fig_ax1 = self.fig.add_subplot(2,2,1)
        self.fig_ax1.set_title("steps / episodes")
        self.fig_ax1.set_xlabel("episodes")
        self.fig_ax1.set_ylabel("steps")
        self.lines_ax1,  = self.fig_ax1.plot([], color='blue')

        # １エピソードの累積報酬のグラフ
        self.fig_ax2 = self.fig.add_subplot(2,2,2)
        self.fig_ax2.set_title("total rewards / episodes")
        self.fig_ax2.set_xlabel("episodes")
        self.fig_ax2.set_ylabel("total rewards")
        self.lines_ax2,  = self.fig_ax2.plot([], color='blue')

        # 平均損失関数の推移グラフ
        self.fig_ax3 = self.fig.add_subplot(2,2,3)
        self.fig_ax3.set_title("loss averages / episodes")
        self.fig_ax3.set_xlabel("episodes")
        self.fig_ax3.set_ylabel("loss averages")
        self.lines_ax3,  = self.fig_ax3.plot([], color='blue')

        # 平均MaxQ値の推移グラフ
        self.fig_ax4 = self.fig.add_subplot(2,2,4)
        self.fig_ax4.set_title("MaxQ averages / episodes")
        self.fig_ax4.set_xlabel("episodes")
        self.fig_ax4.set_ylabel("MaxQ averages")
        self.lines_ax4,  = self.fig_ax4.plot([], color='blue')

    def update(self, episodes, step, total_reward, avg_loss, avg_MaxQ): # 毎エピソード呼び出されるグラフの情報更新関数（やりようによっては、一定間隔で描画更新されるグラフも作れる）
        # リストに情報追加
        self.cnt_episodes.append(episodes)
        self.steps_per_episodes.append(step)
        self.total_rewards.append(total_reward)
        self.avg_loss_history.append(avg_loss)
        self.avg_MaxQ_history.append(avg_MaxQ)

        # 許容量以上のデータを保存した場合、古いものから削除
        if len(self.cnt_episodes) > self.datastock :
            self.cnt_episodes.pop(0)
            self.steps_per_episodes.pop(0)
            self.total_rewards.pop(0)
            self.avg_loss_history.pop(0)
            self.avg_MaxQ_history.pop(0)

    def dataset(self, episodes): # グラフの描画更新（データのセット）
        self.lines_ax1.set_data(self.cnt_episodes, self.steps_per_episodes)
        self.fig_ax1.set_xlim((max([1, episodes - self.datastock]), episodes))
        self.fig_ax1.set_ylim((np.array(self.steps_per_episodes).min(), np.array(self.steps_per_episodes).max() + 0.01))

        self.lines_ax2.set_data(self.cnt_episodes, self.total_rewards)
        self.fig_ax2.set_xlim((max([1, episodes - self.datastock]), episodes))
        self.fig_ax2.set_ylim((np.array(self.total_rewards).min(), np.array(self.total_rewards).max() + 0.01))

        self.lines_ax3.set_data(self.cnt_episodes, self.avg_loss_history)
        self.fig_ax3.set_xlim((max([1, episodes - self.datastock]), episodes))
        self.fig_ax3.set_ylim((np.array(self.avg_loss_history).min(), np.array(self.avg_loss_history).max() + 0.01))

        self.lines_ax4.set_data(self.cnt_episodes, self.avg_MaxQ_history)
        self.fig_ax4.set_xlim((max([1, episodes - self.datastock]), episodes))
        self.fig_ax4.set_ylim((np.array(self.avg_MaxQ_history).min(), np.array(self.avg_MaxQ_history).max() + 0.01))

    def plot(self): # 最終的なグラフを描画する
        self.dataset(self.cnt_episodes[len(self.cnt_episodes) - 1])
        plt.show()

    def is_ready(self): # 描画開始できる状態かどうかの判定
        return self.ready_flag




# デモ用の環境クラス（じゃんけんカードゲーム）
class DemoEnvironment:
    def __init__(self, n_player, n_enemy, match):
        self.n_player = n_player # プレイヤーのカードの枚数（0以下のときは手を自由に選べる）
        self.n_enemy = n_enemy # 敵のカードの枚数

        self.cnt_match = 0
        self.match = match # １エピソードの試合数
        self.te = ["グー  ", "チョキ", "パー  "] # 表示用の文字列配列
        self.card = [] # 場にあるカード
        self.sel_player = -1 # プレイヤーの選択したカード
        self.sel_enemy = -1 # 敵の選択したカード
        self.hand_player = -1 # プレイヤーの選択した手
        self.hand_enemy = -1 # 敵の選択した手

    def update(self, action): # 行動を元に処理の更新を行う（戻り値：次の状態, 報酬, 終了判定）
        #選んだカードの決定
        self.sel_player = action
        self.sel_enemy = rng.integers(self.n_enemy)

        #出した手の決定
        if self.n_player > 0 : self.hand_player = self.card[self.sel_player]
        else : self.hand_player = self.sel_player
        self.hand_enemy = self.card[self.sel_enemy + self.n_player]

        #勝敗の判定（これを報酬にする。勝ち：+1、引分：±0、負け：-1）
        result = (self.hand_enemy - self.hand_player + 4) % 3 - 1

        #カードの入れ替え（全て交換）
        self.card = [rng.integers(3) for i in range(self.n_player + self.n_enemy)]

        #終了判定（指定の試合数で終了）
        self.cnt_match += 1
        if self.cnt_match==self.match : flag = True
        else : flag = False

        return self.card, result, flag

    def render(self): # 描画処理を行う（あまり丁寧には作っていない）
        print(f'自分:{self.te[self.hand_player]} | 相手:{self.te[self.hand_enemy]}', end='') # 出した手の表示
        if self.cnt_match==self.match : print()
        else : print(f' | 次の手:{[self.te[self.card[i]] for i in range(self.n_player + self.n_enemy)]}') # 次のカードの表示

    def reset(self): # ゲームの処理のリセットを行う（戻り値：最初の状態）
        self.cnt_match = 0
        self.card = [rng.integers(3) for i in range(self.n_player + self.n_enemy)]
        return self.card


# メイン関数（デモ環境で学習を行うだけ）
def main():
    #細かい設定
    DISPLAY = False # 学習中の環境の方の描画を行うかどうか
    EPISODE_NUM = 1004 # 学習を行うエピソード数
    GRAPH_INTERVAL = 100 # グラフを描画する間隔（episode）
    GRAPH_DATASTOCK = 10000 # グラフ用に保存しておくデータの最大数(episode)。これを超えると、古いものからデータを削除していく。
    n_player = 3 # プレイヤー側の手札数
    n_enemy = 3 # 敵側の手札数

    #エージェントと環境の用意
    agent = Agent(n_player + n_enemy, n_player + (3 - n_player) * (n_player <= 0))
    env = DemoEnvironment(n_player, n_enemy, 40)

    #学習開始
    agent.init_plot(GRAPH_INTERVAL, GRAPH_DATASTOCK) # グラフ描画準備
    for ep in range(EPISODE_NUM):
        #エピソードの始まり
        done = False # 終了判定をFalseに
        state_next = env.reset() # 環境を初期状態にする
        while not done:
            state = state_next # 状態を更新
            action = agent.select_action(state, True) # エージェントの行動選択
            state_next, reward, done = env.update(action) # 処理の更新と報酬の獲得
            if DISPLAY : env.render() # 画面描画
            agent.train(state, action, reward, state_next, done) # エージェントの学習
    #print(f'steps:{agent.graph.steps_per_episodes} | total reward:{agent.graph.total_rewards} | loss:{agent.graph.avg_loss_history} | MaxQ:{agent.graph.avg_MaxQ_history}')
    agent.show()


if __name__ == '__main__' :
    main()
