s8 = "Hi He Lied Because Boron Could Not Oxidize Fluorine. New Nations Might Also Sign Peace Security Clause. Arthur King Can."

words = s8.split()
first_char_list = [1, 5, 6, 7, 8, 9, 15, 16, 19]
ans = []

for i in range(len(words)):
    if i + 1 in first_char_list:
        ans.append(words[i][0])
    else:
        ans.append(words[i][0:2])

print(ans)