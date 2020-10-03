import os
import nltk

# nltk.download()
# import nltk.corpus
# from nltk.book import gutenberg

# print(os.listdir(nltk.data.find("corpora")))

# nltk.corpus.gutenberg.fileids()

# hamlet=nltk.corpus.gutenberg.words('shakespeare-hamlet.txt')

# print(hamlet)

# for word in hamlet[:500]:
#     print(word,sep=' ',end=' ')

AI="Artificial intelligence algorithms are designed to make decisions, often using real-time data. They are unlike passive machines that are capable only of mechanical or predetermined responses. Using sensors, digital data, or remote inputs, they combine information from a variety of different sources, analyze the material instantly, and act on the insights derived from those data. With massive improvements in storage systems, processing speeds, and analytic techniques, they are capable of tremendous sophistication in analysis and decisionmaking."

from nltk.tokenize import word_tokenize
AI_Tokens= word_tokenize(AI)
# print(AI_Tokens)
# print(len(AI_Tokens))

# from nltk.probability import FreqDist
# fdist=FreqDist()

# for word in AI_Tokens:
#     fdist[word.lower()]+=1
    # print(fdist)

# fdist_top10=fdist.most_common(10)
# print(fdist_top10)   

# from nltk.tokenize import blankline_tokenize
# AI_Blank= blankline_tokenize(AI)
# print(len(AI_Blank))

from nltk.util import bigrams,trigrams,ngrams
string = "Topic sentences are similar to mini thesis statements. Like a thesis statement, a topic sentence has a specific main point. Whereas the thesis is the main point of the essay, the topic sentence is the main point of the paragraph. Like the thesis statement, a topic sentence has a unifying function. But a thesis statement or topic sentence alone doesn’t guarantee unity. An essay is unified if all the paragraphs relate to the thesis, whereas a paragraph is unified if all the sentences relate to the topic sentence. Note: Not all paragraphs need topic sentences. In particular, opening and closing paragraphs, which serve different functions from body paragraphs, generally don’t have topic sentences."
quote_token=nltk.word_tokenize(string)

quotes_bigram= list(nltk.bigrams(quote_token))
# print(quotes_bigram)


quotes_tigram= list(nltk.trigrams(quote_token))
# print(quotes_tigram)


quotes_ngram= list(nltk.ngrams(quote_token,4))
# print(quotes_ngram)


from nltk.stem import PorterStemmer
pst=PorterStemmer()
# print(pst.stem("Having"))


# words_to_steam=["give","giving","given","gave"]
# for words in words_to_steam:
#     print( words ,":", pst.stem(words))


from nltk.stem import LancasterStemmer
lst=LancasterStemmer()
print(lst.stem("Having"))


# words_to_steam=["give","giving","given","gave"]
# for words in words_to_steam:
#     print( words ,":", lst.stem(words))


from nltk.stem import wordnet
from nltk.stem import WordNetLemmatizer
word_len=WordNetLemmatizer()

# for words in words_to_steam:
#     print(words ,":",word_len.lemmatize(words))


from nltk.corpus import stopwords
# print(stopwords.words('english'))

# print(len(stopwords.words('english')))

# import re
# punctuation=re.compile('[]-,?!,:;()|0-9]')
# post_punctuation=[]
# for words in AI_Tokens:
#     words=punctuation.sub("",words)
#     if(len(words) > 0):
#         post_punctuation.append(words)
# print(post_punctuation)

sent ="Timothy is a natural when it comes tp drawing"
# sent_token=word_tokenize(sent)

# for token in sent_token:
#     print(nltk.pos_tag([token]))


sent ="Jhon is eating a delicious cake"
# sent_token=word_tokenize(sent)

# for token in sent_token:
#     print(nltk.pos_tag([token]))


from nltk import ne_chunk

# NE_SENT="The Us President Stays in The White House"

# NE_Token=word_tokenize(NE_SENT)
# NE_Tag=nltk.pos_tag(NE_Token)
# NE_NER=ne_chunk(NE_Tag)
# print(NE_NER)



NE_SENT="The Us President Stays in The White House"

NE_Token=word_tokenize(NE_SENT)
NE_Tag=nltk.pos_tag(NE_Token)

grammer_np=r"NP: {<DT>?<JJ>*<NN>]"

chunk_parser=nltk.RegexpParser(grammer_np)
# chunk_result=chunk_parser(new_token)
# print(chunk_result)