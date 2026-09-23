import numpy as np

#入力の4x2行列Xと正解の4x1行列Tを作成
X = np.array([[0,0],[0,1],[1,0],[1,1]])
T = np.array([[0],[1],[1],[0]])

#隠れ層の重みとバイアスを初期化。とりあえず[-1,1]、0にしてみる
W1 = np.random.uniform(-1.0, 1.0, (2, 2))
b1 = np.array([[0,0]])
#出力層の重みとバイアスを初期化
W2 = np.random.uniform(-1.0, 1.0, (2, 1))
b2 = np.array([[0]])
#学習率を適当に設定
eta = 0.1

np.set_printoptions(suppress=True) #結果が見やすいように指数表記をしない

#学習ループの回数の設定
num = 10000

#学習ループ
for i in range(num):

    #順伝播
    z1 = X @ W1 + b1
    h = 1 / (1 + np.exp(-z1))
    z2 = h @ W2 + b2
    y = 1 / (1 + np.exp(-z2))
    if i == 0 or i == num -1:
        print(y)

    #損失関数（交差エントロピー誤差）
    ##バッチ学習のために独立なデータの同時尤度（積）を対数化した場合は和になる。ただし、データ数が変わってもパラメータが大きく変動して学習が不安定にならないよう、1/Nで影響が減るようにしたほうがよい。和をNで割るということはすなわち平均を取ることになる。
    E = np.mean( -( T * np.log(y) + (1 - T) * np.log(1 - y) ) )

    #逆伝播
    delta2 = y - T #出力層のエラー。4x1行列
    dW2 = ( h.T @ delta2 ) / len(X) #x1δ1+...+x4δ4を出すことで全体に適した勾配を計算。
    db2 = np.mean(delta2, axis=0, keepdims=True) #行方向にだけ平均を取る（縦を潰す）ことでバイアスの次元を崩さないようにする
    ##誤差逆伝播法により誤差を隠れ層に戻す
    delta1 = delta2 @ W2.T * (h * (1 - h))
    dW1 = X.T @ delta1 / len(X)
    db1 = np.mean(delta1, axis=0, keepdims=True)
    ##値を更新
    W2 = W2 - eta * dW2
    b2 = b2 - eta * db2
    W1 = W1 - eta * dW1
    b1 = b1 - eta * db1
