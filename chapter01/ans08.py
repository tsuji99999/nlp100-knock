def cipher(text: str):
    ans = ""
    for char in text:
        if 97 <= ord(char) <= 122:
            ans += chr(219 - ord(char))
        else:
            ans += char
    return ans

s10 = "I have a pen."
print(cipher(s10))