from nltk.tokenize import sent_tokenize
import string

num_files = 4912
write_file = "nsfw.txt"
csv_file = open(write_file,'w')
x,y = 0,0
for i in range(1,num_files):
    f = open(f'data/file_{i}.txt','r',encoding='utf-8', errors='ignore')
    content = f.read()
    tokens = content.split('.')
    if i%100==0:
        print(i)
    for s in tokens:
        s.encode('utf-8').strip()
        s = " ".join(s.split())
        for j in string.punctuation:
            s = s.replace(j,'')
        s = s.strip()
        s = s.lower()
        s = str(s)
        if len(s)<=15:
            y+=1
            continue
        csv_file.write(s+'\n')
        x+=1
csv_file.close()
print(y/x)