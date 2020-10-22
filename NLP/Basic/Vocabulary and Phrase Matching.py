import spacy
nlp=spacy.load("en_core_web_sm")
from spacy.matcher import Matcher
matcher=Matcher(nlp.vocab)

# pattern1=[{'LOWER':'solarpower'}]
# pattern2=[{'LOWER':'solar'},{'LOWER':'power'}]
# pattern3=[{'LOWER':'solar'},{'IS_PUNCT':True},{'LOWER':'power'}]

# matcher.add('Solarpower',None,pattern1,pattern2,pattern3)

doc=nlp(u'The Solar Power industry continues to grow as demand \ for solarpower increases. Solar-Power cars  are gaining popularity.')
# found_match=matcher(doc)
# print(found_match)
# for match_id,start,end in found_match:
#     string=nlp.vocab.strings[match_id]
#     span=doc[start:end]
#     print(match_id,string,start,end,span.text)

# pattern1=[{'LOWER':'solarpower'}]
# pattern2=[{'LOWER':'solar'},{'IS_PUNCT':True,'OP':'*'},{'LEMMA':'power'},{'LOWER':'power'}]
# matcher.remove('SolarPower')
# matcher.add('SolarPower',None,pattern1,pattern2)
# found_matches=matcher(doc)
# print(found_matches)


# pattern1=[{'LOWER':'solarpower'}]
# pattern2=[{'LOWER':'solar'},{'IS_PUNCT':True,'OP':'*'},{'LEMMA':'power'},{'LOWER':'power'}]
# pattern3=[{'LOWER':'solarpowered'}]
# pattern4=[{'LOWER':'solar'},{'IS_PUNCT':True,'OP':'*'},{'LEMMA':'power'},{'LOWER':'power'}]
# matcher.remove('SolarPower')
# matcher.add('SolarPower',None,pattern1,pattern2,pattern3,pattern4)
# found_matches=matcher(doc)
# print(found_matches)
import string
from spacy.matcher import PhraseMatcher
matcher=PhraseMatcher(nlp.vocab,attr='LOWER')
terms=['Galaxy Note','Iphone 11','Iphone X1','Google Pixel']
patterns=[nlp(text) for text in terms]
matcher.add("TerminologyList",None,*patterns)
text_doc=nlp("Unlike Apple, Google isn't shy about sharing about its plans for Pixel 4. The company recently revealed Google Pixel 4 will be the company's first smartphone to run on Soli, a radar-based motion sensing chip.Samsung will launch Galaxy Note 10 on August 7. Unlike previous years, Samsung will introduce multiple iterations of Galaxy Note 10, giving it a series-like treatment")
match=matcher(text_doc)
print(match)

match_id,start,end =match[1]
print(nlp.vocab.strings[match_id],text_doc[start:end])
