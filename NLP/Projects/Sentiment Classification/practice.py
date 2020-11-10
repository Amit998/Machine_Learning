import spacy
from  spacy import displacy
from spacy.lang.en.stop_words import STOP_WORDS

nlp=spacy.load('en_core_web_sm')


# text="This is first sentence. This is another one.Here The 3rd one, Apple and Google"

# doc=nlp(text)

# sent=nlp.create_pipe('sentencizer')
# nlp.add_pipe(sent,before='parser')


# doc=nlp(text)

# for token in doc.sents:
#     print(token)



# stopwords=list(STOP_WORDS)
# print(stopwords)


# for token in doc:
    # if(token.is_stop ==False):
        # print(token)


#### Lemmatization

# doc=nlp('run runs running runner')


# for lem in doc:
#     print(lem.text,lem.lemma_)


## POS


# doc=nlp('All Is Well at your end')


# for token in doc:
#      print(token.text,token.pos_)

# displacy.render(doc,style='dep')


## Entity Detection

# doc =nlp("At a time when it seems that film animation has been dominated by Disney/Pixar's CGI masterpieces, it is both refreshing and comforting to know that Miyazaki is still relying on traditional hand-drawn animation to tell his charming and enchanting stories. ")

# displacy.render(doc,style='ent')

## Text Classification
