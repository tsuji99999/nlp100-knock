import string

s7 = "Now I need a drink, alcoholic of course, after the heavy lectures involving quantum mechanics."

for p in list(string.punctuation):
    s7 = s7.replace(p, "")

words = s7.split()
words_length = []

for word in words:
    words_length.append(len(word))

print(words_length)