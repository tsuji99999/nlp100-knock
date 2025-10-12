import random

def typoglycemia(text: str):
    words = text.split()
    for i, word in enumerate(words):
        if len(word) > 4:
            middle = list(word[1:-1])
            random.shuffle(middle)
            shuffled = "".join(middle)
            words[i] = word[0] + ''.join(shuffled) + word[-1]
    
    return ' '.join(words)

s11 = "I couldn’t believe that I could actually understand what I was reading : the phenomenal power of the human mind ."
print(typoglycemia(s11))