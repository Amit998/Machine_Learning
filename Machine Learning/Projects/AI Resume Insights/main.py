import PyPDF2

# resume_file='DXC Resume Format.pdf'
resume_file="Amit's_Updated_2021_02.pdf"

filehandle=open(f'D:/study/CV/{resume_file}','rb')


pdfReader=PyPDF2.PdfFileReader(filehandle)
pageHandle=pdfReader.getPage(0)
text=pageHandle.extractText()
text=text.replace('\n', '')
text=text.replace('|','')
text=text.replace('  ','')

print(text)

from ibm_watson import PersonalityInsightsV3
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator, authenticator


apiKey=''
url=''


authenticator=IAMAuthenticator(apiKey)
personality_insights=PersonalityInsightsV3(
    version="2017-10-13",
    authenticator=authenticator
)

personality_insights.set_service_url(url)

profile=personality_insights.profile(text,accept='application/json').get_result()

# print(profile)
# print(profile.keys)

for personality in profile['personality']:
    print(personality['name'],personality['percentile'])


for personality in profile['values']:
    print(personality['name'], personality['percentile'])


for personality in profile['needs']:
    print(personality['name'], personality['percentile'])


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

needs=profile['needs']
result={need['name']:need['percentile'] for need in needs}
df=pd.DataFrame.from_dict(result,orient='index')
df.reset_index(inplace=True)
df.columns=['needs','percentile']

# df.head()

plt.figure(figsize=(15,5))
sns.barplot(y='percentile',x='need',data=df).set_title("Traits By Percentile")
plt.show()
