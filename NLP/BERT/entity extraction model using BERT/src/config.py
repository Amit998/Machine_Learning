import transformers
from transformers import pipeline
unmasker = pipeline('fill-mask', model='bert-base-uncased')
print(unmasker("Hello I'm a [MASK] model."))

MAX_LEN=128
TRAIN_BATCH_SIZE=32
VALID_BATCH_SIZE=8
EPOCHS=10
BASE_MODEL_PATH="../input/bert_base_uncases"
MODEL_PAT="model.bin"
TRAINING_FILE="../input/ner_dataset.csv"
TOKENIZER=transformers.BertTokenizer.from_pretrained(
    BASE_MODEL_PATH,
    do_lower_case=True,
)