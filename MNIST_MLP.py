import numpy as np
import matplotlib.pyplot as plt

num = 20 #学習ループ数
eta = 0.1 #学習率。とりあえず定数
n = 128 #隠れ層のノード数
batch_size = 128 #ミニバッチのサイズ

#トレーニングデータを読み込む
##https://github.com/cvdfoundation/mnistからダウンロード、gzを展開した状態で保存しておく。

##画像データの読み込み
with open("MNIST/train-images-idx3-ubyte", 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8, offset=16) #データの形状に合わせただけ
    data = data.reshape(60000, 784) #一繋がりのデータを行列化
    X1 = data / 255.0 #これで各値が0~1に正規化される
##ラベルデータの読み込み
with open("MNIST/train-labels-idx1-ubyte", 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    T = np.eye(10)[data] #単位行列のラベルデータに含まれる行目を取り出す=ラベルデータと同じところだけ1で他が0のOne-Hot行列が作れる

#多層パーセプトロンのための値の初期化。隠れ層はZ1=X1W+b1で、X1が60000x784、Z1が60000xn行列になるため、重み行列W1は784xn、バイアスは1xn。出力層はZ2=Z1W+b2で、Z1が60000xn、Z2が60000x10行列になるため、重み行列W2はnx10、バイアスは1x10
W1 = np.random.randn(784, n) * np.sqrt(2 / 784) #隠れ層にReLUを使うのでHeの初期値にしてみる
b1 = np.random.randn(1, n) * 0.01
#出力層の重みとバイアスを初期化
W2 = np.random.randn(n, 10) * np.sqrt(2 / n)
b2 = np.random.randn(1, 10) * 0.01

#学習ループ
for epoch in range(num):
    narabi = np.random.permutation(60000)
    X1_shuffled = X1[narabi] #適当に並び替えた行列を作る
    T_shuffled = T[narabi]

    for i in range(0, 60000, batch_size): #並び替えたところから(バッチサイズ,784)を切り取り
        X1_batch = X1_shuffled[i:i+batch_size]
        T_batch = T_shuffled[i:i+batch_size]

        ##順伝播の計算
        ###隠れ層
        Z1 = X1_batch @ W1 + b1
        X2 = np.maximum(0, Z1)
        ###出力層
        Z2 = X2 @ W2 + b2
        Z2_max = np.max(Z2, axis=1, keepdims=True) #各行の最大値を取り出す
        exp_Z2 = np.exp(Z2 - Z2_max) #全体から最大値を引いておく。これでe^xのxが大きくなることによるオーバーフローを防ぐ
        Y = exp_Z2 / np.sum(exp_Z2, axis=1, keepdims=True) #合計が1になるようにする

        ##損失の計算
        E = -(1 / batch_size)*np.sum(T_batch * np.log(Y + 1e-15)) #正解クラスが出力される確率ykから確率質量関数がΠn k=1 yk^tkになる。独立同分布の過程を置けば、負の平均対数尤度はE=-(1/N)logL=-(1/N)(Σ N i=1 Σ 10 k=1 tik*log(yik)になる。ただし、yikが0のときlogが-∞に飛んでしまうため、10^-15を足してゼロにならないようにする。
        if E < 0.01:
        #    print(str(i)+"回で早期終了")
            break
        #print("["+str(i)+"/"+str(num)+"]"+str(E))

        #逆伝播
        delta2 = (1 / batch_size) * (Y - T_batch) #エラーを事前にNで割っておいたもの。
        dW2 = ( X2.T @ delta2 ) #重みWの勾配を計算。
        db2 = np.sum(delta2, axis=0, keepdims=True) #行方向にだけ平均を取る（縦を潰す）ことでバイアスの次元を崩さないようにする。事前に平均を取ったのでmean→sumに変更
        ##誤差逆伝播法により誤差を隠れ層に戻す
        delta1 = ( delta2 @ W2.T ) * (Z1 > 0)  #ReLUの微分は不連続で定義できないが、>0のとき1,<=0のとき0として、前者は誤差の値をそのまま戻し、後者は誤差自体戻さない。 60000x10行列と10xn行列の積で60000xn行列
        dW1 = X1_batch.T @ delta1
        db1 = np.sum(delta1, axis=0, keepdims=True)
        ###値を更新
        W2 = W2 - eta * dW2
        b2 = b2 - eta * db2
        W1 = W1 - eta * dW1
        b1 = b1 - eta * db1

#結果を確認する

##テストデータの読み込み
with open("MNIST/t10k-images-idx3-ubyte", 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    data = data.reshape(10000, 784)
    X1_test = data / 255.0
with open("MNIST/t10k-labels-idx1-ubyte", 'rb') as f:
    Y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

##順伝播で計算
Z1_test = X1_test @ W1 + b1
X2_test = np.maximum(0, Z1_test)
Z2_test = X2_test @ W2 + b2
result = np.argmax(Z2_test, axis=1)

correct = 0
for i in range(10000):
    if result[i] == Y_test[i]:
        correct += 1
print("正答率：" + str((correct / 10000) * 100) + "%")

while True:
    index = input("index? (1~10000): ")
    plt.imshow(data[int(index)].reshape(28, 28), cmap="gray")
    plt.title("Label: "+str(Y_test[int(index)])+" Estimated: "+str(result[int(index)]))
    plt.show()
