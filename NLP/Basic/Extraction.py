from re import match

import spacy
from spacy.matcher import Matcher, matcher
from spacy.tokens import Span
from spacy import displacy



nlp=spacy.load('en_core_web_sm')



matcher=Matcher(nlp.vocab)


# pattern=[
#     {
#         "LOWER":"facebook"
#     },
#     {
#         "LEMMA":"be"
#     },
#     {
#         "POS":"ADV",
#         "OP":"*"
#     },
#     {
#         "POS":"ADJ"
#     }
# ]

# matched_Sent=[]
# doc=nlp('i like to say facebook is devil. from devices with Internet connectivity, such as personal computers, tablets and smartphones. After registering, users can create a profile revealing information about themselves. facebook is even more worst than we have thought')





# def callBack_method_fb(matcher,doc,i,matches):
#     matched_id,start,end=matches[i]
#     span=doc[start:end]
#     sent=span.sent

#     match_ent=[{
#         'start':span.start_char - sent.start_char,
#         'end':span.end_char - sent.start_char,
#         'label':'MATCH'
#     }]

#     # print(match_ent)

#     matched_Sent.append({'text':sent.text,'ents':match_ent})

# matcher.add("fb",callBack_method_fb,pattern)


# matches=matcher(doc)
# print(matched_Sent)


# pattern=[
#     {
#         "ORTH":"("
#     },
#     {
#         "IS_DIGIT":True
#     },
#     {
#         "ORTH":")"
#     },
    
#     {
#         "IS_DIGIT":True
#     },

#     {
#         "ORTH":"-",
#         "OP":"?"
#     },

#     {
#         "IS_DIGIT":True
#     },
# ]


# doc=nlp('Call me At (91) 45989 - 45493')

# # print([t.text for t in doc])
# matcher.add("PhoneNumber",None,pattern)
# matches=matcher(doc)
# print(matches)

# for match_id,start,end in matches:
#     span=doc[start:end]
#     print(span.text)


# For Email



# pattern=[
#     {
#         "TEXT":{
#             "REGEX":"[a-zA-Z0-9-_.]+@[a-zA-Z0-9-_.]"
#         }
#     }
# ]


# doc=nlp('My Email ID dbjhb@gmail.com and damii@gmail.com')

# # print([t.text for t in doc])
# matcher.add("Email",None,pattern)
# matches=matcher(doc)
# print(matches)

# for match_id,start,end in matches:
#     span=doc[start:end]
#     print(span.text)

# pos_emoji= [ "😀" , "😃", "😄" ,"😁" ,"😆" ,"😅" ,"😂" ,"🤣" ]
# neg_emoji=["😣" ,"😖", "😫", "😩", "🥺", "😢", "😭", "😤", "😠" ,"😡", "🤬", "🤯"]

# pos_pattern=[[{"ORTH" :emoji}] for emoji in pos_emoji]
# neg_pattern=[[{"ORTH" :emoji}] for emoji in neg_emoji]

# # print(pos_pattern)


# def label_sentiment(matcher,doc,i,matches):
#     match_id,start,end=matches[i]
#     if(doc.vocab.strings[match_id] == 'HAPPY'):
#         doc.sentiment += 0.1
#     elif (doc.vocab.strings[match_id] == 'SAD'):
#         doc.sentiment -=0.1


# matcher=Matcher(nlp.vocab)
# matcher.add("HAPPY",label_sentiment,*pos_pattern)
# matcher.add("SAD",label_sentiment,*neg_pattern)

# matcher.add("HASTAG",None,[{"TEXT":"#"},{"IS_ASCII":True}])
# doc=nlp("Helloword 😄 😭  #AMITDUTTA ")
# matches=matcher(doc)

# for match_id,start,end in matches:
#     print(doc.vocab.strings[match_id])
#     span=doc[start:end]
#     print(span.text)


# Efficent Phrase matching


# from spacy.matcher import PhraseMatcher

# matcher=PhraseMatcher(nlp.vocab)

# terms=["BARAC OBAMA","MODI","RAJNIKANT"]

# pattern = [nlp.make_doc(text) for text in terms]
# print(pattern)


# matcher.add('term',None, *pattern)

# doc=nlp("barack Obama, MODI indian PM. South Star RAJNIKANT")

# matches=matcher(doc)


# for match_id,start,end in matches:
#     print(doc.vocab.strings[match_id])
#     span=doc[start:end]
#     print(span.text)

from spacy.pipeline import EntityRuler
nlp=spacy.load('en_core_web_sm')


patterns=[
    {"label":"ORG","pattern":"AMIT DUTTA"},
    {"label":"GPE","pattern":[{"LOWER":"los"},{"LOWER":"ANGELS"}]}
]

ruler=EntityRuler(nlp)
ruler.add_patterns(patterns)
nlp.add_pipe(ruler)

doc=nlp("Amit DUTTA AMIT DUTTA is a simple boy and he wanted to go to los ANGELS")

for ent in doc.ents:
    print(ent.text,ent.label_)