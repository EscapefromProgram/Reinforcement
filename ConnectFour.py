import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl
import numpy as np


# ゲームボード
class Board():
    def reset(self):
        self.board = np.array([[0] * 7 for _ in range(6)], dtype=np.float32)
        self.winner = None
        self.missed = False
        self.done = False
        self.placed = np.array([5] * 7)

    def move(self, act, turn):
        if self.placed[act] >= 0:
            self.board[self.placed[act]][act] = turn
            self.placed[act] -= 1
            self.check_winner(act, turn)
        else:
            self.winner = turn*-1
            self.done = True
            self.missed = True
        # if self.placed[act] < 0:
        #     self.missed = True
        #     act = self.get_empty_pos()
        #
        # self.board[self.placed[act]][act] = turn
        # self.placed[act] -= 1
        # self.check_winner(act, turn)

        if np.count_nonzero(self.board) == 42:
            self.winner = 0
            self.done = True

    def check_winner(self, act, turn):
        # 縦│ 横─ 右下＼ 右上／ check
        if self.connected(act, turn, 1, 0) >= 3 or self.connected(act, turn, 0, 1) >= 3 or \
                self.connected(act, turn, 1, 1) >= 3 or self.connected(act, turn, -1, 1) >= 3:
            self.winner = turn
            self.done = True
            return

    # 特定の方向に石の連結が存在するか
    def connected(self, act, turn, step_i, step_j):
        i = self.placed[act] + 1
        j = act
        count = 0
        for _ in range(2):
            index_i = step_i
            index_j = step_j
            while 0 <= (i + index_i) < 6 and \
                    0 <= (j + index_j) < 7:
                if self.board[i + index_i][j + index_j] != turn:
                    break
                else:
                    count += 1
                    index_i += step_i
                    index_j += step_j
            if count >= 3:
                return count
            step_i *= -1
            step_j *= -1
        return count

    def get_empty_pos(self):
        empties = np.where(self.placed >= 0)[0]
        if len(empties) > 0:
            return np.random.choice(empties)
        else:
            return 0

    def show(self):
        row = " {} | {} | {} | {} | {} | {} | {} "
        hr = "\n---------------------------\n"
        nr = "\n===========================\n"
        tempboard = []
        for i in np.ravel(self.board):
            if i == 1:
                tempboard.append("o")
            elif i == -1:
                tempboard.append("×")
            else:
                tempboard.append(" ")
        # for i in range(7):
        #     if self.placed[i] >= 0:
        #         tempboard[self.placed[i] * 7 + i] = i + 1
        num = " 1 | 2 | 3 | 4 | 5 | 6 | 7"
        print((row + hr + row + hr + row + hr + row + hr +
               row + hr + row + nr + num + "\n").format(*tempboard))


# explorer用のランダム関数オブジェクト
class RandomActor:

    def __init__(self, board):
        self.board = board
        self.random_count = 0

    def random_action_func(self):
        self.random_count += 1
        return self.board.get_empty_pos()


# Q関数
class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels=294):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_hidden_channels),
            l3=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):
        # -1を扱うのでleaky_reluとした
        h = F.elu(self.l0(x))
        h = F.elu(self.l1(h))
        h = F.elu(self.l2(h))
        return chainerrl.action_value.DiscreteActionValue(self.l3(h))


# ボードの準備
b = Board()
# explorer用のランダム関数オブジェクトの準備
ra = RandomActor(b)
# 環境と行動の次元数
obs_size = 42
n_actions = 7
# Q-functionとオプティマイザーのセットアップ
q_func = QFunction(obs_size, n_actions)
optimizer = chainer.optimizers.Adam(eps=1e-2)
optimizer.setup(q_func)
# 報酬の割引率
gamma = 0.98
# Epsilon-greedyを使ってたまに冒険。50000ステップでend_epsilonとなる
explorer = chainerrl.explorers.LinearDecayEpsilonGreedy(
    start_epsilon=1.0, end_epsilon=0.05, decay_steps=30000, random_action_func=ra.random_action_func)
# Experience ReplayというDQNで用いる学習手法で使うバッファ
replay_buffer = chainerrl.replay_buffer.ReplayBuffer(capacity=10 ** 6)
# Agentの生成（replay_buffer等を共有する2つ）
agent_p1 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100)
agent_p2 = chainerrl.agents.DoubleDQN(
    q_func, optimizer, replay_buffer, gamma, explorer,
    replay_start_size=500, update_interval=1,
    target_update_interval=100)

# 学習ゲーム回数
n_episodes = 0
# カウンタの宣言
miss = 0
win = [0, 0]
draw = 0
# エピソードの繰り返し実行
for i in range(1, n_episodes + 1):
    b.reset()
    reward = 0
    agents = [agent_p1, agent_p2]
    turn = np.random.choice([0, 1])
    last_state = None
    while not b.done:
        # 配置マス取得
        action = agents[turn].act_and_train(b.board.copy(), reward)
        # 配置を実行
        b.move(action, 1)
        # 配置の結果、終了時には報酬とカウンタに値をセットして学習
        if b.done:
            if b.winner == 1:
                reward = 1
                win[turn] += 1
            elif b.winner == 0:
                draw += 1
            else:
                reward = -1
            if b.missed:
                miss += 1
                reward += -1
            # エピソードを終了して学習
            agents[turn].stop_episode_and_train(b.board.copy(), reward, True)
            # 相手もエピソードを終了して学習。相手のミスは勝利として学習しないように
            if agents[1 if turn == 0 else 0].last_state is not None and b.missed is False:
                # 前のターンでとっておいたlast_stateをaction実行後の状態として渡す
                agents[1 if turn == 0 else 0].stop_episode_and_train(last_state, reward*-1, True)
        else:
            # 学習用にターン最後の状態を退避
            last_state = b.board.copy()
            # 継続のときは盤面の値を反転
            b.board = b.board * -1
            # ターンを切り替え
            turn = 1 if turn == 0 else 0

    # コンソールに進捗表示
    if i % 100 == 0:
        print("episode:", i, " / rnd:", ra.random_count, " / miss:", miss,
              " / win_0:", win[0], " / win_1:", win[1], " / draw:", draw, " / statistics:",
              agent_p1.get_statistics(), " / epsilon:", agent_p1.explorer.epsilon)
        # カウンタの初期化
        miss = 0
        win = [0, 0]
        draw = 0
        ra.random_count = 0
    if i % n_episodes == 0:  # モデルを保存
        agent_p1.save("ox4_result5")

print("Training finished.")


# agent_p1.load("ox4_result4")  # DL agent


# 人間のプレーヤー
class HumanPlayer:
    def act(self, board):
        valid = False
        while not valid:
            try:
                act = int(input("Please enter 1-7: "))
                if 1 <= act <= 7 and board[b.placed[act - 1]][act - 1] == 0:
                    valid = True
                    return act-1
                else:
                    print("Invalid move")
            except Exception as e:
                print(act + " is invalid")

# 検証
human_player = HumanPlayer()
while True:
    next = int(input("CPUと対戦しますか？ はい : 1 or いいえ : 0  "))
    if next != 1:
        break
    b.reset()
    dqn_first = np.random.choice([True, False])
    while not b.done:
        # DQN
        if dqn_first or np.count_nonzero(b.board) > 0:
            b.show()
            action = agent_p1.act(b.board.copy())
            b.move(action, 1)
            if b.done:
                b.show()
                if b.winner == 1:
                    print("DQN Win")
                elif b.winner == 0:
                    print("Draw")
                else:
                    print("DQN Missed")
                agent_p1.stop_episode()
                continue
        # 人間
        b.show()
        action = human_player.act(b.board.copy())
        b.move(action, -1)
        if b.done:
            # b.show()
            if b.winner == -1:
                print("HUMAN Win")
            elif b.winner == 0:
                print("Draw")
            agent_p1.stop_episode()

print("Test finished.")

