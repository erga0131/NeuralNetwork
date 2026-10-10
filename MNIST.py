import numpy as np
import matplotlib.pyplot as plt

num = 50 #学習ループ数
eta = 0.1 #学習率。とりあえず定数

#トレーニングデータを読み込む
##https://github.com/cvdfoundation/mnistからダウンロード、gzを展開した状態で保存しておく。

##画像データの読み込み
with open("MNIST/train-images-idx3-ubyte", 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8, offset=16) #データの形状に合わせただけ
    data = data.reshape(60000, 784) #一繋がりのデータを行列化
    X = data / 255.0 #これで各値が0~1に正規化される
##ラベルデータの読み込み
with open("MNIST/train-labels-idx1-ubyte", 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8, offset=8)
    T = np.eye(10)[data] #単位行列のラベルデータに含まれる行目を取り出す=ラベルデータと同じところだけ1で他が0のOne-Hot行列が作れる

#単層パーセプトロンのための値の初期化。Z=XW+bで、Xが60000x784、Zが60000x10行列になるため、重み行列Wは784x10
W = np.random.randn(784, 10) * np.sqrt(1 / 784) #Xavierの初期値（の簡易版）
b = np.random.randn(1, 10) * 0.01 #バイアスは適当に小さい値で

#学習ループ
for i in range(num):
    ##順伝播の計算
    Z = X @ W + b
    Z_max = np.max(Z, axis=1, keepdims=True) #各行の最大値を取り出す
    exp_Z = np.exp(Z - Z_max) #全体から最大値を引いておく。これでe^xのxが大きくなることによるオーバーフローを防ぐ
    Y = exp_Z / np.sum(exp_Z, axis=1, keepdims=True) #合計が1になるようにする

    ##損失の計算
    E = -(1 / 60000)*np.sum(T * np.log(Y + 1e-15)) #正解クラスが出力される確率ykから確率質量関数がΠn k=1 yk^tkになる。独立同分布の過程を置けば、負の平均対数尤度はE=-(1/N)logL=-(1/N)(Σ N i=1 Σ 10 k=1 tik*log(yik)になる。ただし、yikが0のときlogが-∞に飛んでしまうため、10^-15を足してゼロにならないようにする。
    if E < 0.01:
        print(str(i)+"回で早期終了")
        break
    print("["+str(i)+"/"+str(num)+"]"+str(E))

    #逆伝播
    delta = (1 / 60000) * (Y - T) #エラーを事前にNで割っておいたもの。60000x10行列。
    dW = ( X.T @ delta ) #重みWの勾配を計算。 784x60000行列と60000x10行列の行列積を取って784x10行列になる。重み行列Wの大きさに一致する。
    db = np.sum(delta, axis=0, keepdims=True) #行方向にだけ平均を取る（縦を潰す）ことでバイアスの次元を崩さないようにする。事前に平均を取ったのでmean→sumに変更
    ###値を更新
    W = W - eta * dW
    b = b - eta * db

#結果を確認する

##テストデータの読み込み
with open("MNIST/t10k-images-idx3-ubyte", 'rb') as f:
    data = np.frombuffer(f.read(), dtype=np.uint8, offset=16)
    data = data.reshape(10000, 784)
    X_test = data / 255.0
with open("MNIST/t10k-labels-idx1-ubyte", 'rb') as f:
    Y_test = np.frombuffer(f.read(), dtype=np.uint8, offset=8)

##順伝播で計算
Z_test = X_test @ W + b
result = np.argmax(Z_test, axis=1)

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
