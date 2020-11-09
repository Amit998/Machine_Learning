import spacy
from spacy.matcher import matcher
from spacy.tokens import Span
from spacy import displacy

nlp=spacy.load('en_core_web_sm')


# def add_title(doc):
#     new_ents=[]
#     for ent in doc.ents:
#         if(ent.label_ =='PERSON' and ent.start != 0):
#             prev_token= doc[ent.start-1]
#             if(prev_token.text in ('Dr','Dr.','Mr','Mr.')):
#                 new_ent=Span(doc,ent.start-1 ,ent.end,label=ent.label)
#                 new_ents.append(new_ent)
#             else:
#                 new_ents.append(ent)
#     doc.ents=new_ents
#     return doc


# nlp.add_pipe(add_title,after='ner')
# doc=nlp('Dr Amit Dutta is the owner of a tea stall named google chai wala')

# print([(ent.text,ent.label_ )for ent in doc.ents])


# #### USE PART OF SPEECH


# displacy.render(doc,style='dep',options={'compact':True,'ditance':100})


# def add_person_orgs(doc):
#     per_ents=[ent for ent in doc.ents if ent.label_ == "PERSON"]

#     for ent in per_ents:
#         head = ent.root.head
#         if(head.lemma_ == 'work'):
#             preps=[token for token in head.children if token.dep_== 'prep']
#             for prep in preps:
#                 orgs=[token for token in prep.children if token.ent_type == 'ORG']
#                 print({'Person':ent,'org':orgs,'past':head.tag_=='VBD'})
#     return doc


from spacy.pipeline import merge_entities






# nlp.add_pipe(merge_entities)
# nlp.add_pipe(add_person_orgs)
# doc=nlp('Amit Dutta worked at Microsoft ')
# for i in doc:
#     print(i.tag_,i.text)
# print(doc)


# Modify Model


# def add_person_orgs(doc):
#     per_ents=[ent for ent in doc.ents if ent.label_ == "PERSON"]
    
#     orgs=[]
#     past=[]
   

#     for ent in per_ents:
        
#         head = ent.root.head
        
#         if(head.lemma_ == 'work'):
           
#             preps=[token for token in head.children if token.dep_== 'prep']


#             for prep in preps:
#                 orgs=[token for token in prep.children if token.ent_type == 'ORG']

#                 aux=[token for token in head.children if token.dep_ == 'aux']
#                 past_aux=any(t.tag_ == 'VBD' for t in aux)
#                 past=head.tag_ == 'VBD' or head.tag_ == 'VBG' and past_aux

#                 # print({'Person':ent,'org':orgs,'past':head.tag_=='VBD'})
#             print({'person':ent,'orgs':orgs,'past':past})
#     return doc


# nlp.add_pipe(merge_entities)
# nlp.add_pipe(add_person_orgs)
# doc=nlp('Amit Dutta was working at Microsoft')



doc=nlp('This is Raw text')

texts=["This is raw text","There is lots of text"]

docs=list(nlp.pipe(texts))