import spacy
import spacy.displacy as displacy


nlp=spacy.load('en_core_web_sm')
# doc=nlp("TensorFlow is an open source software library for high performance numerical computation. Its flexible architecture allows easy deployment of computation across a variety of platforms (CPUs, GPUs, TPUs), and from desktops to clusters of servers to mobile and edge devices.")

# doc=nlp(u'Jhon cena')

# doc=nlp("Jhon Cena")

doc = nlp(u'Apple is looking at buying U.K. startup for $1 billion')


# def show_ents(doc):
#     if doc.ents:
#         for ent in doc.ents:
#             print(ent.text +' - '+ ent.label_+' - '+ str(spacy.explain(ent.label_)))
#     else:
#         print('No Named Entities Found.')

# show_ents(doc)

for ent in doc.ents:
    print(ent.text,ent.start_char,ent.end_char,ent.label_)