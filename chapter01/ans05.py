def n_gram(n: int, text: str):
    return [text[idx:idx+n] for idx in range(len(text)- n + 1)]

s9 = "I am an NLPer"
print(n_gram(3,s9))

words = s9.split()
print(n_gram(2, words))