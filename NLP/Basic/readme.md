## Lemmatization:-Lemmatization technique is like stemming. The output we will get after lemmatization is called ‘lemma’, which is a root word rather than root stem, the output of stemming. After lemmatization, we will be getting a valid word that means the same thing.


## What is Stemming?
## Stemming is a technique used to extract the base form of the words by removing affixes from them. It is just like cutting down the branches of a tree to its stems. For example, the stem of the words eating, eats, eaten is eat.


## StopWords
## In computing, stop words are words which are filtered out before or after processing of natural language data (text).[1] Though "stop words" usually refers to the most common words in a language, there is no single universal list of stop words used by all natural language processing tools, and indeed not all tools even use such a list.


## Ruled-Based Matching

## Token-Based Matching



## ORTH	unicode	The exact verbatim text of a token.

## TEXT 	unicode	The exact verbatim text of a token.

## LOWER	unicode	The lowercase form of the token text.

## LENGTH	int	The length of the token text.

## IS_ALPHA, IS_ASCII, IS_DIGIT	    bool	Token text consists of alphabetic characters, ASCII characters, digits.

## IS_LOWER, IS_UPPER, IS_TITLE	bool	Token text is in lowercase, uppercase, titlecase.

## IS_PUNCT, IS_SPACE, IS_STOP	bool	Token is punctuation, whitespace, stop word.

## IS_SENT_START	bool	Token is start of sentence.

## LIKE_NUM, LIKE_URL, LIKE_EMAIL	bool	Token text resembles a number, URL, email.

## POS, TAG, DEP, LEMMA, SHAPE	unicode	The token’s simple and extended part-of-speech tag, dependency label, lemma, shape.

## ENT_TYPE	unicode	The token’s entity label.

## _ V2.1	dict	Properties in custom extension attributes.


## Named Entity Recognition