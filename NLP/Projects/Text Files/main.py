name="Amit Dutta"

print('My Name Is {}'.format(name))

print(f'My Name Is {name}')


data_science_tuts=[
    ('Python For Beginners',19),
    ('Feature Selecting for ML',19),
    ('Mechine learning',19),
    ('Deep Learning',10),
]


for info in data_science_tuts:
    print(f'{info[0]:{50}} {info[1]:{10}} ')

# >,^

for info in data_science_tuts:
    print(f'{info[0]:^{50}} {info[1]:{10}} ')


# .csv and .tsv



import pandas as pd

data=pd.read_csv('SMSSpamCollection.tsv',sep='\t')

print(data.shape)

import PyPDF2 as pdf

# file=open('AI_Topics.pdf','rb')

file=open('Eligibility Criteria_Year 2021_Engineering_v1.0.pdf','rb')



# print(file)

pdf_reader=pdf.PdfFileReader(file)

# help(pdf)

print(pdf_reader.getIsEncrypted())

print(pdf_reader.getNumPages())

page=pdf_reader.getPage(0)
print(page.extractText())


### Append Write or Merge PDFs


pdf_writer=pdf.PdfFileWriter()

pdf_writer.addPage(page)
pdf_writer.addPage(page)

output=open('Eligibility Criteria_Year 2021_Engineering_v1.0.pdf','wb')
pdf_writer.write(output)
output.close()

# %%writefile test.txt
# hello, this is NLP Lesson
# Please Do Contact me


