from transformers import pipeline
from bs4 import BeautifulSoup
import requests


summarizer=pipeline("summarization")

URL='https://hackernoon.com/how-i-write-and-send-out-newsletters-that-get-almost-50percent-open-rates-and-less-than-05percent-unsubscribes-r7r3663'

r=requests.get(URL)
# print(r.text)

soup=BeautifulSoup(r.text,'html.parser')
results=soup.find_all(['h1','p'])

# print(result)

text =[ result.text for result in results ]

# print(text)
ARTICLE=' '.join(text)
# print(ARTICLE)

ARTICLE=ARTICLE.replace('.','.<eos>')
ARTICLE=ARTICLE.replace('?','?<eos>')
ARTICLE=ARTICLE.replace('!','!<eos>')
sentences=ARTICLE.split('.<eos>')


# print(sentences[1])

max_chunk=500
current_chunk=0
chunks=[]


for sentence in sentences:
    if (len(chunks) == current_chunk +1):
        if len(chunks[current_chunk]) + len(sentence.split(' ')) <= max_chunk:
            chunks[current_chunk].extend(sentence.split(' '))
        else:
            current_chunk +=1
            chunks.append(sentence.split(' '))
    else:
        print(current_chunk)
        chunks.append(sentence.split(' '))

# print(len(chunks[0]))

for chunk_id in range(len(chunks)):
    chunks[chunk_id]=' '.join(chunks[chunk_id])

# print(chunks)

res=summarizer(chunks,max_length=120,min_length=30,do_sample=False)

# print(res)

summary=' '.join(summ['summary_text'] for summ in res)


print(summary)

with open('summary.txt','w') as f:
    f.write(summary)