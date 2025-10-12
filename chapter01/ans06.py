def n_gram(n: int, text: str):
    return [text[idx:idx+n] for idx in range(len(text)- n + 1)]

x = "paraparaparadise"
y = "paragraph"

X = n_gram(2, x)
Y = n_gram(2, y)

# 和集合
print("和集合: ", list(set(X) | set(Y)))

# 積集合
print("積集合: ", list(set(X) & set(Y)))

# 差集合
print("差集合: ", list(set(X) - set(Y)))

# "se"という文字列がXおよびYに含まれるか
print("\"se\"が含まれる？")
print("X: ", "se" in X)
print("Y: ", "se" in Y)