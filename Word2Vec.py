import numpy as np
import re #正規表現のために使う

#データの準備

filename = "sentence.txt"

with open(filename, 'r') as text_original:
    text2 = text_original.readlines()

with open(filename, 'r') as text_original:
    text = text_original.read()
    text = text.lower() #Theとtheを同じ単語と認識できるよう小文字化
    text = re.sub(r'[^a-z\s]', ' ', text) #正規表現でアルファベットと空白文字（改行・スペース等）以外を半角スペースに
    text = re.sub(r'\s+', ' ', text) #正規表現で連続する空白文字を単スペースに置き換え

text = text.split() #単語ごとにリスト化
vocab = list(set(text)) #語彙をリスト化。順序は関係ないのでsetで重複を消す。後に辞書にする関係でlistに戻す

word2id = dict() #ここで語彙にidをつけて辞書化（idを振りたいだけなので特に順番は気にせず、上の出力順）
for i in range(len(vocab)):
    word2id[vocab[i]] = i

#id2word = dict(); j = 0 #可視化に使うので逆引き辞書を作成
#for i in word2id:
#    id2word[j] = i
#    j += 1
#古いバージョンは順番が保持されずunsafeらしいのでちゃんとした実装を調べた
id2word = {v: k for k, v in word2id.items()}

###print(word2id)
###print(id2word)

#入力と正解のペアを作る
##Skip-gramの実装なので、入力がi(i=1,...,textの単語数-1)動く時、それを入力にして正解が前の単語と後ろの単語になるよう、textの単語数x2から最初の単語の前と最後の単語の後ろの2つを引いた行数と、2列で行列を作成。
"""
matrix = np.zeros((len(text) * 2 - 2, 2)); j = 0
for i in range(1, len(text) - 1):
    matrix[j, 0] = word2id[text[i]]
    matrix[j, 1] = word2id[text[i - 1]]
    matrix[j + 1, 0] = word2id[text[i]]
    matrix[j + 1, 1] = word2id[text[i + 1]]
    j += 2
#最初の単語の後ろと最後の単語の前が含まれていないので末尾に追加してしまう
matrix[j, 0] = word2id[text[0]]
matrix[j, 1] = word2id[text[1]]
matrix[j + 1, 0] = word2id[text[len(text) - 2]]
matrix[j + 1, 1] = word2id[text[len(text) - 1]]
"""

#ウィンドウが1は厳しかったので長さを可変にしてみる
window = 2
pairs = []
for i in range(len(text2)):
    ###一文ごとに単語ごとリスト化
    words2 = text2[i].lower() #Theとtheを同じ単語と認識できるよう小文字化
    words2 = re.sub(r'[^a-z\s]', ' ', words2) #正規表現でアルファベットと空白文字（改行・スペース等）以外を半角スペースに
    words2 = re.sub(r'\s+', ' ', words2)
    words2 = words2.split()
    #print(str(i)+" / "+str(len(text2)))
    for j in range(len(words2)):
        center = word2id[words2[j]]
        for w in range(1, window + 1):
            if j - w >= 0:
                pairs.append((center, word2id[words2[j - w]]))
            if j + w < len(words2):
                pairs.append((center, word2id[words2[j + w]]))

matrix = np.array(pairs)

#print(matrix)

#学習する
##MNIST_MLP.pyを流用する。
##変更点：バイアスは使わないので削除。入力の次元をvに変更。入力をX1とTからmatrixに変更、活性化関数を使わないため初期値を正規分布に戻す、逆伝播の微分でReLUを通さないようにする。バッチ処理内のX1,TをOne-Hot表現を通したものに変更
v = len(vocab)

num = 500 #学習ループ数
eta = 0.005 #学習率
n = 8 #埋め込み次元数
batch_size = 10 #ミニバッチのサイズ

#Adamのための初期化
##移動平均（一次モーメント）で急な変化を軽減（Momentum）
m_W1 = np.zeros((v, n))
m_W2 = np.zeros((n, v))
##分散の移動平均（二次モーメント）で勾配の大きさによって学習率を調整していく（RMSProp）
v_W1 = np.zeros((v, n))
v_W2 = np.zeros((n, v))
##外生的に決定する値
t = 0 #学習ステップ数の保持
beta_1 = 0.9 #一次モーメントの減衰率（これくらいが目安らしい）
beta_2 = 0.999 #二次モーメントの減衰率（同上）

#多層パーセプトロンのための値の初期化。
W1 = np.random.randn(v, n) * 0.01
#出力層の重みを初期化
W2 = np.random.randn(n, v) * 0.01

#学習ループ
N = matrix.shape[0]

for epoch in range(num):
    narabi = np.random.permutation(N)
    X1_shuffled = matrix[:,0][narabi] #適当に並び替えた行列を作る
    T_shuffled = matrix[:,1][narabi]

    print("["+str(epoch)+"/"+str(num)+"]")

    for i in range(0, N, batch_size): #並び替えたところから(バッチサイズ,784)を切り取り
        X1_batch = X1_shuffled[i:i+batch_size]
        T_batch = T_shuffled[i:i+batch_size]

        ##順伝播の計算
        ###隠れ層

        #One-Hotを毎回作ると死ぬほど重かったので作り置きしておく
        #X1 = np.eye(v)[X1_batch.astype(int)]　#まだ重かったので不要な計算を消す
        T = np.eye(v)[T_batch.astype(int)]

        #Z1 = X1 @ W1
        X2 = W1[X1_batch.astype(int)]

        ###出力層
        Z2 = X2 @ W2
        Z2_max = np.max(Z2, axis=1, keepdims=True) #各行の最大値を取り出す
        exp_Z2 = np.exp(Z2 - Z2_max) #全体から最大値を引いておく。これでe^xのxが大きくなることによるオーバーフローを防ぐ
        Y = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True) #合計が1になるようにする

        ##損失の計算
        E = -(1 / len(X1_batch))*np.sum(T * np.log(Y + 1e-15)) #正解クラスが出力される確率ykから確率質量関数がΠn k=1 yk^tkになる。独立同分布の過程を置けば、負の平均対数尤度はE=-(1/N)logL=-(1/N)(Σ N i=1 Σ 10 k=1 tik*log(yik)になる。ただし、yikが0のときlogが-∞に飛んでしまうため、10^-15を足してゼロにならないようにする。
        print(E)

        #逆伝播
        delta2 = (Y - T) / len(X1_batch) #エラーを事前にNで割っておいたもの。
        dW2 = ( X2.T @ delta2 ) #重みWの勾配を計算。
        ##誤差逆伝播法により誤差を隠れ層に戻す
        delta1 = ( delta2 @ W2.T )
        dW1 = np.zeros_like(W1) #全て0の勾配の初期状態を作成し、誤差を加算していく。+=ではなくnp.add.atを使うのは2回足す場合を正しく処理できるようにするため。
        np.add.at(dW1, X1_batch.astype(int), delta1)

        ##Adamの実装
        t += 1
        ###一次モーメントの更新
        m_W2 = beta_1 * m_W2 + (1 - beta_1) * dW2
        m_W1 = beta_1 * m_W1 + (1 - beta_1) * dW1
        ###二次モーメントの更新
        v_W2 = beta_2 * v_W2 + (1 - beta_2) * (dW2 ** 2)
        v_W1 = beta_2 * v_W1 + (1 - beta_2) * (dW1 ** 2)
        ###学習初期はほぼ0になってしまうので、カウントtとベータの値を使って補正。tが進むほど分母は1に近くなり値が大きくなる（最初は小さくなる）
        m_W2_hat = m_W2 / (1 - beta_1 ** t)
        m_W1_hat = m_W1 / (1 - beta_1 ** t)
        v_W2_hat = v_W2 / (1 - beta_2 ** t)
        v_W1_hat = v_W1 / (1 - beta_2 ** t)

        ###値を更新
        W2 = W2 - eta * (m_W2_hat / (np.sqrt(v_W2_hat) + 1e-15))
        W1 = W1 - eta * (m_W1_hat / (np.sqrt(v_W1_hat) + 1e-15))

#結果を確認する
while True:
    target_word = input("Target Word? (ex.dog): ")
    id = word2id[target_word]
    result = dict()
    for i in range(len(vocab)):
        if id != i:
            cos_theta = np.dot(W1[id], W1[i]) / (np.linalg.norm(W1[id]) * np.linalg.norm(W1[i]))
            result[vocab[i]] = cos_theta
    topnums = 5 ###類似度上位何単語表示するか
    result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True)[:topnums])
    i = 0
    for key, value in result.items():
        print("["+str(i + 1)+" / "+str(topnums)+"] Word: "+key+" ("+str(value)+")")
        i += 1
