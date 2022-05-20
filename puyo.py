#――最低限の機能だけを付けた、とことん専用ぷよぷよ――
#※ぷよをカサカサ動かすと永遠に地面に着かないようにすることが可能など、拙い部分も若干あり。

# ライブラリをインポート
import tkinter as tk # GUIライブラリ
import random
import copy



#――定数（いじれば色々改造できる）――
# 画面構成設定
WINDOW_WIDTH = 400 # ウィンドウを開いたときの横幅
WINDOW_HEIGHT = 500 # ウィンドウを開いたときの縦幅
WINDOW_PADDING = 25 # キャンバスなどの余白幅

BLOCK_SIZE = 32 # ぷよの縦横サイズ
FIELD_WIDTH = 6 # フィールドの幅
FIELD_HEIGHT = 12 # フィールドの高さ

# ゲームシステム定数
PUYO_COLOR_NUM = 4 # ぷよの色数
PUYO_COLOR = ["whitesmoke", "red", "blue", "green", "yellow", "purple"] # ぷよの色(空, 色1, 色2, 色3, ...)　（基本の５色を用意しているが、PUYO_COLOR_NUMと合わせて変えてもいい）
GAMEOVER_X = 2 # ゲームオーバー判定（×マーク）のｘ座標　（複数箇所には非対応）
ERASE_CONNECT = 4 # ぷよの消える最低連結数（通常４）

DRAW_CONTROL = True # 操作ぷよを動かす様子を描写するかどうか
DRAW_DETAIL = True # 操作ぷよが接地してから次のツモを引くまでの流れの詳細を描写するかどうか
DRAW_SYSTEM = True # 操作ぷよが接地してから次のツモを引くまでの流れの大まかな流れを描写するかどうか

# ウェイト秒数
PUT_WAIT = 1000 # 操作ぷよが自由落下するのにかかる時間(ms)
FALL_WAIT = 50 # 設置した後や連鎖した後、空中にあるぷよが１マス下に自由落下するのに掛かる時間(ms)
REN_WAIT = 800 # 連鎖後の待機秒数(ms)

# システム定数（特に変更する必要はなし。逆に変更すると不具合が起こる可能性大）
MOVE_UP = 0 # 上方向移動を示す定数
MOVE_RIGHT = 1 # 右方向移動を示す定数
MOVE_DOWN = 2 # 下方向移動を示す定数
MOVE_LEFT = 3 # 左方向移動を示す定数
ROLL_RIGHT = 4 # 右回転を示す定数
ROLL_LEFT = 5 # 左回転を示す定数



#・設定カスタムセット。最初に関数を呼び出しておくと設定が変わる。今回は使ってないけど。
def tibipuyo_preset():
    #ちびぷよカスタム
    global BLOCK_SIZE, FIELD_WIDTH, FIELD_HEIGHT, GAMEOVER_X, PUT_WAIT, FALL_WAIT, REN_WAIT
    BLOCK_SIZE = 20
    FIELD_WIDTH = 10
    FIELD_HEIGHT = 18
    GAMEOVER_X = 4
    PUT_WAIT = 700
    FALL_WAIT = 20
    REN_WAIT = 100

def dekapuyo_preset():
    #でかぷよカスタム
    global WINDOW_WIDTH, BLOCK_SIZE, FIELD_WIDTH, FIELD_HEIGHT, GAMEOVER_X, PUT_WAIT
    WINDOW_WIDTH = 450
    BLOCK_SIZE = 64
    FIELD_WIDTH = 3
    FIELD_HEIGHT = 6
    GAMEOVER_X = 1
    PUT_WAIT = 1200



#ぷよを構成するクラス
class PuyoBlock:
    def __init__(self, color=PUYO_COLOR[0]): # １つのぷよ（正方形）を作成
        self.color = color

    def set_color(self, color): # ぷよの色を設定
        self.color = color

    def get_color(self): # ぷよの色を取得
        return self.color


#フィールドに積まれたぷよを管理するクラス
class PuyoField:
    def __init__(self):
        self.width = FIELD_WIDTH
        self.height = FIELD_HEIGHT

        # フィールド初期化（ぷよは画面上部にも１段分隠れて積まれているので、１段余分にリストを用意）
        self.blocks = [PuyoBlock(PUYO_COLOR[0]) for x in range(self.width) for y in range(self.height + 1)]
        self.ren = 0

    def set_ren(self, ren): # 連鎖数を設定
        self.ren = ren

    def get_ren(self): # 連鎖数を取得
        return self.ren

    def get_width(self): # フィールドの横幅を取得
        return self.width

    def get_height(self): # フィールドの縦幅を取得
        return self.height

    def get_blocks(self): # フィールドを管理するリストを取得
        return self.blocks

    def get_block(self, x, y): # 指定座標のフィールド情報を取得
        return self.blocks[y * self.width + x]

    def is_gameover(self): # ゲームオーバーかの判定
        return self.get_block(GAMEOVER_X, 1).get_color() != PUYO_COLOR[0]

    def is_can_move(self, puyo, move_direction=None, roll_direction=None): # 指定の場所に操作ぷよが動けるかの判定
        # 指定の状態にする（引数のpuyoが変更されないように、新たにPuyoControlを作成する）
        test_puyo = PuyoControl()
        x, y = puyo.get_coord()
        test_puyo.set_coord(x, y)
        test_puyo.set_angle(puyo.get_angle())
        if move_direction is not None : test_puyo.move(move_direction)
        if roll_direction is not None : test_puyo.roll(roll_direction)

        # 判定の流れ
        #　 壁にめり込んでいないかの判定
        #　→画面上部に切れていた場合、画面上部に隠れているぷよの有無で判定
        #　→移動場所に既にぷよが存在しているかの判定
        px, py = test_puyo.get_coord()
        if px < 0 or px >= self.width or py > self.height :
            return False
        elif py < 0 :
            if self.get_block(px, 0).get_color() != PUYO_COLOR[0] :
                return False
        elif self.get_block(px, py).get_color() != PUYO_COLOR[0] :
            return False

        px2, py2 = test_puyo.calc_subcoord(px, py, test_puyo.get_angle())
        if px2 < 0 or px2 >= self.width or py2 > self.height :
            return False
        elif py2 < 0 :
            if self.get_block(px2, 0).get_color() != PUYO_COLOR[0] :
                return False
        elif self.get_block(px2, py2).get_color() != PUYO_COLOR[0] :
            return False

        return True

    def put_puyo(self, puyo): # 操作ぷよを設置してフィールドに反映
        px, py = puyo.get_coord() # 操作ぷよの位置を取得
        color = puyo.get_color(0) # 操作ぷよの色を取得
        if py >= 0 :
            self.get_block(px, py).set_color(PUYO_COLOR[color[0]])

        px2, py2 = puyo.calc_subcoord(px, py, puyo.get_angle())
        if py2 >= 0 :
            self.get_block(px2, py2).set_color(PUYO_COLOR[color[1]])

    def fall(self): # フィールド全体のぷよを１マス自由落下させる
        flag = False
        for y in range(self.height - 1, -1, -1):
            for x in range(self.width):
                if self.get_block(x, y).get_color() != PUYO_COLOR[0] and self.get_block(x, y + 1).get_color() == PUYO_COLOR[0] :
                    self.get_block(x, y + 1).set_color(self.get_block(x, y).get_color())
                    self.get_block(x, y).set_color(PUYO_COLOR[0])
                    flag = True

        return flag # 落下が発生したかどうかをTorFで返す

    def reset_connect_count(self): # connect_countで用いる、探索完了したかどうかを示す配列を初期化する
        self.finish_flag = [[False for y in range(self.height + 1)] for x in range(self.width)]

    def connect_count(self, x, y, count=0, flag=True): # 連結数を数える再帰関数（引数のcountは0から、flagはTrueからスタートすること）
        # flagがTrueのとき（探索開始したとき）、変数を初期化する
        if flag :
            # 探索開始場所にぷよがないとき、連結数0として返す
            if self.get_block(x, y).get_color() == PUYO_COLOR[0] :
                self.finish_flag[x][y] = True
                return 0

            # 別の箇所と連結していて既に探索完了していた場合は、探索省略して-1を返す
            elif self.finish_flag[x][y] : return -1

            # 探索したことがあるかどうかを示す配列を初期化
            self.checked_flag = [[0 for y in range(self.height + 1)] for x in range(self.width)]

        # まず、現在位置に探索済のフラグを立てる
        self.checked_flag[x][y] = True

        # 周囲探索開始
        cx, cy = (0, -1)
        for i in range(4):
            # 周囲の未探索の同色ぷよがあるかの調査
            if x + cx >= 0 and x + cx < FIELD_WIDTH and y + cy > 0 and y + cy <= FIELD_HEIGHT :
                if not( self.checked_flag[x+cx][y+cy] ) and self.get_block(x+cx, y+cy).get_color() == self.get_block(x, y).get_color() :
                    # 次の探索場所の調査を開始
                    count = self.connect_count(x + cx, y + cy, count, flag=False)

                # 探索終了したら、そこに探索済のフラグを立てる
                self.checked_flag[x + cx][y + cy] = True

            # 調査場所移動（回転行列より90度回転）
            cx, cy = -cy, cx

        # 周囲の探索が終了したら、count+1を戻り値として返す
        self.finish_flag[x][y] = True
        return count + 1

    def erase_connect(self, x, y, flag=True): # 連結しているぷよを全て消去する再帰関数
        # flagがTrueのとき（探索開始したとき）、変数を初期化する
        if flag :
            # 探索開始場所にぷよがないとき、何もせずに終了する
            if self.get_block(x, y).get_color() == PUYO_COLOR[0] : return

            # 探索済エリアの記憶配列を作成
            self.checked_area2 = [[False for y in range(self.height + 1)] for x in range(self.width)]

        # まず、現在位置に探索済のフラグを立てる
        self.checked_area2[x][y] = True

        # 周囲探索開始
        cx, cy = (0, -1)
        for i in range(4):
            # 周囲の未探索の同色ぷよがあるかの調査
            if x + cx >= 0 and x + cx < FIELD_WIDTH and y + cy > 0 and y + cy <= FIELD_HEIGHT :
                if not( self.checked_area2[x+cx][y+cy] ) and self.get_block(x+cx, y+cy).get_color() == self.get_block(x, y).get_color() :
                    # 次の探索場所の調査を開始
                    self.erase_connect(x + cx, y + cy, flag=False)

                # 探索終了したら、そこは終了フラグを立てる
                self.checked_area2[x + cx][y + cy] = True

            # 調査場所移動（回転行列より90度回転）
            cx, cy = -cy, cx

        # 周囲の探索が終了したら、自分のいる場所のぷよを消去して戻る
        self.get_block(x, y).set_color(PUYO_COLOR[0])
        return


# 操作するぷよを管理するクラス
class PuyoControl:
    def __init__(self):
        # 操作ぷよの色をネクストぷよ含めて作成する
        self.count_slot = 0 # 何個目の操作ぷよかを数える
        self.colorslot = [(random.randrange(1, PUYO_COLOR_NUM + 1), random.randrange(1, PUYO_COLOR_NUM + 1)) for i in range(3)]

        # 操作ぷよの状況を初期化
        self.exist = False # 操作中かどうかのフラグ
        self.x = GAMEOVER_X
        self.y = 1
        self.angle = MOVE_UP # 回転方向

    def set_coord(self, x, y): # 操作ぷよ（軸ぷよ）の座標を設定
        self.x = x
        self.y = y

    def get_coord(self): # 操作ぷよ（軸ぷよ）の座標を取得
        return self.x, self.y

    def set_angle(self, angle): # 操作ぷよの回転方向を設定
        self.angle = angle

    def get_angle(self): # 操作ぷよの回転方向を取得
        return self.angle

    def get_countslot(self): # 何個目の操作ぷよかを取得する
        return self.count_slot

    def get_colorslot(self): # 操作ぷよ・ネクストスロットのぷよの色のリストを取得
        return self.colorslot

    def set_color(self, point, color1, color2): # 指定の位置のぷよの色を設定
        self.colorslot[point] = (color1, color2)

    def get_color(self, point): # 指定の位置のぷよの色を取得
        return self.colorslot[point]

    def set_exist(self, flag): # 操作中かのフラグを設定
        self.exist = flag

    def is_exist(self): # 操作中かどうかを判定
        return self.exist

    def push_color(self): # 操作ぷよの色を更新する（操作ぷよの色のスロットの一つずらし、ネクネクぷよの色を決める）
        self.count_slot += 1
        self.colorslot.pop(0)
        self.colorslot.append((random.randrange(1, PUYO_COLOR_NUM + 1), random.randrange(1, PUYO_COLOR_NUM + 1)))

    def calc_subcoord(self, x, y, angle): # 操作ぷよ（軸ではない方のぷよ）の座標を計算
        if angle == MOVE_UP :
            return x, y - 1
        elif angle == MOVE_RIGHT :
            return x + 1, y
        elif angle == MOVE_DOWN :
            return x, y + 1
        else :
            return x - 1, y

    def move(self, direction): # 指定の方向に移動させる
        if direction == MOVE_UP :
            self.y -= 1
        elif direction == MOVE_RIGHT :
            self.x += 1
        elif direction == MOVE_DOWN :
            self.y += 1
        elif direction == MOVE_LEFT :
            self.x -= 1

    def roll(self, direction): # 指定の方向に回転させる
        if direction == ROLL_RIGHT :
            self.angle = (self.angle + 1) % 4
        else :
            self.angle = (self.angle + 3) % 4


# 画面描画を管理するキャンバスクラス
class PuyoCanvas( tk.Canvas ):
    def __init__(self, root, field): # 描画するキャンバスの作成
        # キャンバスの大きさ計算
        canvas_width = (field.get_width() + 2) * BLOCK_SIZE # キャンバスの横幅は、（フィールドの横幅）＋（余白１マス）＋（ネクストぷよ分１マス）
        canvas_height = field.get_height() * BLOCK_SIZE

        # tk.Canvasのinit
        super().__init__(root, width=canvas_width, height=canvas_height, bg="white")

        # キャンバスを設置
        self.place(x=WINDOW_PADDING, y=WINDOW_PADDING)

        # 正方形を描画して画面作成
        self.block = [[] for x in range(field.get_width())]
        for y in range(field.get_height()):
            for x in range(field.get_width()):
                block = field.get_block(x, y + 1)
                x1 = x * BLOCK_SIZE
                x2 = (x + 1) * BLOCK_SIZE
                y1 = y * BLOCK_SIZE
                y2 = (y + 1) * BLOCK_SIZE
                self.block[x].append(
                    self.create_rectangle(
                        x1, y1, x2, y2,
                        outline="gray", width=1,
                        fill=block.get_color()
                    )
                )

        # ネクストぷよ描画（現段階では空）
        self.nextblock = []
        for i in range(2) :
            for y in range(2) :
                self.nextblock.append(
                    self.create_rectangle(
                        (field.get_width() + 1) * BLOCK_SIZE, (y + 3 * i) * BLOCK_SIZE, (field.get_width() + 2) * BLOCK_SIZE, (y + 3 * i + 1) * BLOCK_SIZE,
                        outline="gray", width=1, fill=PUYO_COLOR[0]
                    )
                )

        # ×マークの描画
        self.create_line(GAMEOVER_X * BLOCK_SIZE, 0, (GAMEOVER_X + 1) * BLOCK_SIZE, BLOCK_SIZE, fill="crimson")
        self.create_line(GAMEOVER_X * BLOCK_SIZE, BLOCK_SIZE, (GAMEOVER_X + 1) * BLOCK_SIZE, 0, fill="crimson")

        # 連鎖数表示用テキスト用意
        self.text = tk.StringVar()
        self.rentext = tk.Label(
            root, font=("", 14, "bold"),
            textvariable=self.text, height=1
        )
        self.rentext.place(x = WINDOW_PADDING, y = BLOCK_SIZE * FIELD_HEIGHT + 2 * WINDOW_PADDING)

        # １つ前のフィールド情報やネクストぷよ情報などを記録
        self.before_field = copy.deepcopy(field)
        self.before_point = -1 # 確実に更新が入る値
        self.before_ren = 0

    def update(self, field, puyo): # 画面を更新する
        # 描画用のフィールド作成（フィールド＋操作ぷよ）
        new_field = PuyoField()
        for y in range(field.get_height() + 1):
            for x in range(field.get_width()):
                color = field.get_block(x, y).get_color()
                new_field.get_block(x, y).set_color(color)

        # 操作しているぷよの情報をフィールドに反映
        if puyo.is_exist() :
            new_field.put_puyo(puyo)

        # 描画用のフィールドでキャンバスに描画
        for y in range(field.get_height()):
            for x in range(field.get_width()):
                # (x, y)座標の色を取得
                new_color = new_field.get_block(x, y + 1).get_color()
                
                # (x, y)座標が前回描画したときから変化していなければ描画しない
                old_color = self.before_field.get_block(x, y + 1).get_color()
                if new_color == old_color :
                    continue

                # フィールドの該当箇所に新たに描画
                x1 = x * BLOCK_SIZE
                x2 = (x + 1) * BLOCK_SIZE
                y1 = y * BLOCK_SIZE
                y2 = (y + 1) * BLOCK_SIZE
                block = self.create_rectangle(
                    x1, y1, x2, y2,
                    outline="gray", width=1, fill=new_color
                )
                self.delete(self.block[x][y])
                self.block[x][y] = block

        # ×マークは、隠れていたのが現れたら再描画
        if new_field.get_block(GAMEOVER_X, 1).get_color() == PUYO_COLOR[0] and self.before_field.get_block(GAMEOVER_X, 1).get_color() != PUYO_COLOR[0] :
            self.create_line(GAMEOVER_X * BLOCK_SIZE, 0, (GAMEOVER_X + 1) * BLOCK_SIZE, BLOCK_SIZE, fill="crimson")
            self.create_line(GAMEOVER_X * BLOCK_SIZE, BLOCK_SIZE, (GAMEOVER_X + 1) * BLOCK_SIZE, 0, fill="crimson")

        # １つ前のフィールド情報を更新
        self.before_field = new_field

        # ネクストぷよ表示が変わっていた場合に描画更新
        if puyo.get_countslot() != self.before_point :
            # 描画更新
            colorslot = puyo.get_colorslot()
            for i in range(2) :
                for y in range(2) :
                    new_block = self.create_rectangle(
                        (field.get_width() + 1) * BLOCK_SIZE, (y + 3 * i) * BLOCK_SIZE, (field.get_width() + 2) * BLOCK_SIZE, (y + 3 * i + 1) * BLOCK_SIZE,
                        outline="gray", width=1, fill=PUYO_COLOR[colorslot[i + 1][1 - y]]
                    )
                    self.delete(self.nextblock[i * 2 + y])
                    self.nextblock[i * 2 + y] = new_block

            # １つ前のネクストぷよ情報を更新
            self.before_point = puyo.get_colorslot()

        # 連鎖数表示も以前と異なっていた際に表示更新
        new_ren = field.get_ren()
        if self.before_ren != new_ren :
            self.before_ren = new_ren
            if new_ren == 0 :
                self.text.set("") # 連鎖数0のときはテキストの内容は特になし
            else :
                self.text.set(str(new_ren) + "連鎖！")



# ぷよぷよというゲームを制御するクラス
# （GameからEventHandllerへの繋がりを持たない代わりに、全ての関数において処理終了後に引数のfunc関数を呼び出すようにしている。使うかは自由。）
class PuyoGame:
    def __init__(self, root): # インスタンス作成
        # フィールド管理リストを初期化
        self.field = PuyoField()

        # 操作ぷよを用意
        self.puyo = PuyoControl()
        self.ren = 0 # 直前の連鎖数を保存する変数

        # ぷよぷよの画面を作成
        self.root = root
        self.canvas = PuyoCanvas(root, self.field)
        self.canvas.update(self.field, self.puyo)

        self.set_func(lambda : None)

    def set_func(self, func=lambda:None): # Gameクラスが一通りの処理を終えた後に呼び出す関数の設定
        self.func = func # ※ぷよぷよゲームで遊びたいだけなら、特に変更は必要ない。必要なときに用いる

    def get_ren(self): # 直前の連鎖数を取得する
        return self.ren

    def start(self, end_func): # ゲームを開始
        # 終了時に呼び出す関数をセット
        self.end_func = end_func

        # フィールド管理リストを初期化
        self.field = PuyoField()
        self.field.set_ren(0)

        # 操作ぷよを追加
        self.new_puyo()

    def new_puyo(self): # 操作ぷよを追加
        # ゲームオーバー判定
        if self.field.is_gameover():
            print("GAMEOVER")
            self.end_func()

        else :
            # 連鎖数初期化
            self.ren = self.field.get_ren()
            self.field.set_ren(0)

            # 操作ぷよを追加
            self.puyo.set_exist(True)
            self.puyo.push_color()
            self.puyo.set_coord(GAMEOVER_X, 1)
            self.puyo.set_angle(MOVE_UP)

            # 描画更新
            if DRAW_SYSTEM : self.canvas.update(self.field, self.puyo)

            self.func() # （２回目以降のツモのときは、このタイミングで操作ぷよの自由落下タイマーを作動させる）

    def move_puyo(self, direction): # 操作ぷよを移動させる
        # 移動できる場合だけ移動させる
        if self.field.is_can_move(self.puyo, move_direction=direction):
            # 操作ぷよを移動
            self.puyo.move(direction)

            # 描画更新
            if DRAW_CONTROL : self.canvas.update(self.field, self.puyo)

            self.func() # （funcに操作ぷよの自由落下タイマー開始関数が入っていた場合は、ここでそれを作動させる）

        # 下方向に移動できなかった場合、操作ぷよをフィールドに固定させる
        elif direction == MOVE_DOWN :
            self.puyo.set_exist(False) # 操作モードから連鎖処理モードに移行
            self.field.put_puyo(self.puyo)

            self.root.after(PUT_WAIT, self.fall_loop) # フィールド落下処理へ移行

        else :
            self.func()

    def roll_puyo(self, direction): # 操作ぷよを回転させる
        angle = self.puyo.get_angle() # 現在の回転方向の取得

        # 単純な回転で移動できる場合
        if self.field.is_can_move(self.puyo, roll_direction=direction):
            # 操作ぷよを回転
            self.puyo.roll(direction)

            # 描画更新
            if DRAW_CONTROL : self.canvas.update(self.field, self.puyo)
        
        # 回転方向と逆側に押し出される形でなら移動できる場合
        elif self.field.is_can_move(self.puyo, move_direction=(11-2*direction+angle)%4, roll_direction=direction):
            # 操作ぷよを移動・回転
            self.puyo.move((11 - 2 * direction + angle) % 4)
            self.puyo.roll(direction)

            # 描画更新
            if DRAW_CONTROL : self.canvas.update(self.field, self.puyo)

        # 左右両方が塞がっていて、一つ上に飛び出すことで移動できる場合
        elif self.field.is_can_move(self.puyo, move_direction=MOVE_UP, roll_direction=direction):
            # 操作ぷよを移動・回転
            self.puyo.move(MOVE_UP)
            self.puyo.roll(direction)

            # 描画更新
            if DRAW_CONTROL : self.canvas.update(self.field, self.puyo)

        self.func()

        #※クイックターンは実装しない
        #※回転できない場合、何もせずに終了

    def fall_loop(self, flag=False): # フィールド上の空中にあるぷよを、地面に着くまで自由落下させる関数
        if self.field.fall() :
            # fall関数でまだ落下させられたときは、描画を更新し、数秒後にもう一度fall_loopを呼び出す
            if DRAW_SYSTEM : self.canvas.update(self.field, self.puyo)
            self.root.after(FALL_WAIT, lambda : self.fall_loop(flag=True))

        elif flag :
            # fall関数で１マス以上落下させた後に連結判定へ移行する場合（さらに数秒待たされる）
            if DRAW_DETAIL : self.canvas.update(self.field, self.puyo)
            self.root.after(PUT_WAIT, self.check_connect)

        else:
            # fall関数では一度も落下させずに連結判定へ移行する場合（硬直せずに即座に次のツモへ）
            self.root.after(0, self.check_connect)

    def check_connect(self): # フィールド上で４連結以上しているぷよがあるか確認し、あれば消去する
        flag = False # 消去の有無フラグ
        self.field.reset_connect_count() # 連結数判定関連の配列の初期化
        for y in range(1, self.field.get_height() + 1):
            for x in range(self.field.get_width()):
                if self.field.connect_count(x, y) >= ERASE_CONNECT : # 連結数の調査
                    self.field.erase_connect(x, y) # 該当箇所の消去
                    flag = True

        if flag : # 連鎖が起こったとき
            # 連鎖数を１増やす
            self.field.set_ren(self.field.get_ren() + 1)
            print(str(self.field.get_ren()) + "連鎖！")

            if DRAW_SYSTEM : self.canvas.update(self.field, self.puyo) # 描画更新

            # 連鎖できたときは、フィールド上のぷよの自由落下へ移行
            self.root.after(REN_WAIT, self.fall_loop)

        else : # 連鎖できなかったときは、次のツモへ移行
            self.new_puyo()

#＜関数呼び出しの流れ＞
#  move_puyo- func
#           \ fall_loop- check_connect- new_puyo- func
#                                start/         \ end_func
#  roll_puyo- func


# 操作などのイベントに応じてぷよぷよを制御するクラス
class EventHandller:
    def __init__(self, root, game):
        self.root = root

        # 制御するゲーム
        self.game = game

        # イベントを定期的に発行するタイマー
        self.timer = None

        # ゲームスタートボタンを設置
        self.button = tk.Button(root, text='開始', command=self.start_event)
        self.button.place(x=WINDOW_PADDING + BLOCK_SIZE * (FIELD_WIDTH + 2) + WINDOW_PADDING, y=WINDOW_PADDING + 5)

    def start_event(self): # 開始ボタンを押されたときの処理
        # ぷよぷよ開始
        self.game.start(self.end_event)
        self.running = True

        # タイマーセット
        self.timer_start()

        # キー入力受付開始
        self.root.bind("<Left>", self.left_key_event)
        self.root.bind("<Right>", self.right_key_event)
        self.root.bind("<Down>", self.down_key_event)
        self.root.bind("<z>", self.z_key_event)
        self.root.bind("<x>", self.x_key_event)

    def end_event(self): # ゲーム終了時の処理
        self.running = False

        # イベント受付を停止
        self.timer_end()
        self.root.unbind("<Left>")
        self.root.unbind("<Right>")
        self.root.unbind("<Down>")
        self.root.unbind("<z>")
        self.root.unbind("<x>")

    def timer_start(self): # タイマーを開始
        self.game.set_func()
        # 実行中のタイマーを停止
        if self.timer is not None :
            self.root.after_cancel(self.timer)

        # ぷよぷよを実行中の場合のみタイマースタート
        if self.running :
            self.timer = self.root.after(1000, self.timer_event)

    def timer_end(self): # タイマーを終了
        if self.timer is not None :
            self.root.after_cancel(self.timer)
            self.timer = None

    def timer_event(self): # タイマーが作動したときに実行（下キー入力）
        self.down_key_event(None)

    def down_key_event(self, event): # 下キーが入力されたときの処理
        if self.game.puyo.is_exist():
            # タイマーを終了する
            self.timer_end()

            # ぷよを下に動かす
            self.game.set_func(self.timer_start) # 処理が終了したときに実行する関数に落下タイマースタートを設定
            self.game.move_puyo(MOVE_DOWN)

    def left_key_event(self, event): # 左キーが入力されたときの処理
        # ぷよを左に動かす
        if self.game.puyo.is_exist() : self.game.move_puyo(MOVE_LEFT)

    def right_key_event(self, event): # 右キーが入力されたときの処理
        # ぷよを右に動かす
        if self.game.puyo.is_exist() : self.game.move_puyo(MOVE_RIGHT)

    def z_key_event(self, event): # ｚキーが入力されたときの処理
        # ぷよを左回転させる
        if self.game.puyo.is_exist() : self.game.roll_puyo(ROLL_LEFT)

    def x_key_event(self, event): # ｘキーが入力されたときの処理
        # ぷよを右回転させる
        if self.game.puyo.is_exist() : self.game.roll_puyo(ROLL_RIGHT)


# アプリケーションクラス
class App(tk.Tk):
    def __init__(self):
        super().__init__()

        # アプリウィンドウの設定
        self.geometry(str(WINDOW_WIDTH) + "x" + str(WINDOW_HEIGHT))
        self.title("ぷよぷよ")

        # ぷよぷよ生成
        game = PuyoGame(self)

        # イベントハンドラー生成
        EventHandller(self, game)


if __name__=="__main__" :
    # GUIアプリ生成
    app = App()
    app.mainloop()
