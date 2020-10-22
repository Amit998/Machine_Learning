import spacy
from spacy.tokens import Span




nlp=spacy.load('en_core_web_sm')
doc=nlp("TensorFlow is an open source software library for high performance numerical computation. Its flexible architecture allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.")

# doc=nlp("India")
# doc=nlp('I borrow 100 rupees from Donald Trump')

# doc=nlp(u'Our Company plans to introduce a new vacuum cleaner.'
# u'If Successful , the vacuum cleaner will be our first product')


# doc=nlp(u'Originally Priced at $27.50, the sweater was marked down to five dollars.')



# def show_ents(doc):
#     if doc.ents:
#         for ent in doc.ents:
#             print(ent.text +' - '+ ent.label_+' - '+ str(spacy.explain(ent.label_)))
#     else:
#         print('No Named Entities Found.')


# from spacy.matcher import  PhraseMatcher
# matcher=PhraseMatcher(nlp.vocab)

# phrase_list=['vacuum cleaner','vacuum-cleaner']
# phrase_patterns=[nlp(text) for text in phrase_list]

# matcher.add('newproduct',None,*phrase_patterns)
# matches=matcher(doc)

# print(matches)

# ORG=doc.vocab.strings[u'PERSON']
# new_ent=Span(doc,0,1,label=ORG)
# doc.ents=list(doc.ents)+[new_ent]

# PROD=doc.vocab.strings[u'PRODUCT']
# new_ent=[Span(doc,match[1],match[2],label=PROD) for match in matches]
# doc.ents=list(doc.ents)+ new_ent

# show_ents(doc)

# print(len([ent for ent in doc.ents if ent.label_=='MONEY']))

# Remove ents forms on whitespace


# def remove_whitespace_entities():
#     doc.ents=[e for e in doc.ents if not e.text.isspace()]
#     return doc
# nlp.add_pipe(remove_whitespace_entities,after='ner')

for chunk in doc.noun_chunks:
    print(chunk.text+' - '+chunk.root.text+' - '+chunk.root.dep_+' - '+chunk.root.head.text)
# print(len(doc.noun_chunks))

from spacy import displacy

displacy.render(doc,style='ent')