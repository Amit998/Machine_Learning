# import spacy


# nlp=spacy.load('en_core_web_sm')
# doc=nlp("TensorFlow is an open source software library for high performance numerical computation. Its flexible architecture allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.")

# for token in doc:
#     print(token.text,token.lemma_,token.pos_,token.tag_,token.dep_,token.shape_,token.is_alpha,token.is_stop)


# import spacy

# nlp = spacy.load("en_core_web_sm")
# doc = nlp("Apple is looking at buying U.K. startup for $1 billion")

# for token in doc:
#     print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_,
#             token.shape_, token.is_alpha, token.is_stop)


import spacy

nlp = spacy.load("en_core_web_sm")
# doc = nlp("Autonomous cars shift insurance liability toward manufacturers")
# doc=nlp(u'I read books on NLP')

# doc=nlp(u'I read book on NLP')
# print(doc.text)

# print(doc[4].text,doc[4].pos_,doc[4].tag_,spacy.explain(doc[4].tag_))

# r=doc[1]

# print(f'{r.text:{10}} {r.pos_:{8}} {r.tag_:{6}} {spacy.explain(r.tag_)}')

doc=nlp(u"The Quick Brown Fox jump over the lazy dog's back")
# pos_counts=doc.count_by(spacy.attrs.POS)
# print(pos_counts)

# print(doc.vocab[83].text)


# for k,v in sorted(pos_counts.items()):
#     print(f'{k}.{doc.vocab[k].text:{5}} :{v}')

# TAG_COUNT=doc.count_by(spacy.attrs.TAG)

# for k,v in sorted(TAG_COUNT.items()):
#     print(f'{k}. {doc.vocab[k].text:{4}} :{v}')

from spacy import displacy

# displacy.render(doc,style='dep',jupyter=False,options={'distance':110})

displacy.serve(doc,style="dep")