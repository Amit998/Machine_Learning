import PyPDF2 as pdf

file=open('sample.pdf','rb')


# print(file)

pdf_file_reader=pdf.PdfFileReader(file)
# print(pdf_file_reader.read())

page1=pdf_file_reader.getPage(0)
page2=pdf_file_reader.getPage(1)
# print(page1.extractText())


pdf_writer=pdf.PdfFileWriter()


pdf_writer.addPage(page1)
pdf_writer.addPage(page2)

output=open('pages.pdf','wb')
pdf_writer.write(output)
output.close()