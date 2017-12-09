import re

CENSORED_WORDS = ['white']

sentence = "The whiteness of the wine was cool"

regex_string = ''
for w in CENSORED_WORDS:
    regex_string+=r'\b' + w + r'\b|'
regex_string = regex_string[0:-1] #remove trailing pipe (|)
regex_pat = re.compile(regex_string, flags=re.IGNORECASE)

print regex_string
print regex_pat


sentence = sentence.replace(regex_pat,'')

print sentence
