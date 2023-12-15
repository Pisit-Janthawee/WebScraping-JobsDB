from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from unidecode import unidecode
import contractions
import re

class TextNormalizer:

    def __init__(self):
        pass

    def lowercase(self, string):
        return str(string).lower()

    def remove_punctuation(self, string):
        tokenizer = RegexpTokenizer(r'\w+')
        string = tokenizer.tokenize(string)
        string = " ".join(string)
        return string

    def remove_stop_words(self, string):
        string = string.split()
        stops = set(stopwords.words("english"))
        string = [w for w in string if not w in stops]
        string = " ".join(string)
        return string

    def stemming(self, string):
        stemmer = PorterStemmer()
        words = string.split()
        stemmed_words = [stemmer.stem(word) for word in words]
        return ' '.join(stemmed_words)

    def lemmatization(self, string):
        lemmatizer = WordNetLemmatizer()

        words = string.split(' ')
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized_words)

    def handle_accents_and_diacritics(self, string):
        return unidecode(string)

    def handle_contractions(self, string):
        expanded_words = []
        for word in string.split():
            expanded_words.append(contractions.fix(word))

        expanded_text = ' '.join(expanded_words)
        return expanded_text

    def handle_special_characters(self, string):
        return re.sub(r'[^a-zA-Z0-9\s]', ' ', string)

    def remove_html(self, string):
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', string)

    def text_normalization(self, string):
        string = self.handle_contractions(string)
        string = self.handle_accents_and_diacritics(string)
        string = self.remove_punctuation(string)
        string = self.remove_stop_words(string)
        string = self.lemmatization(string)
        string = self.handle_special_characters(string)
        string = self.lowercase(string)
        return string
     # Others

    def globalization(self, string: str):
        return str(string).replace(';', ' ')

    def remove_html(self, string):
        html_pattern = re.compile('<.*?>')
        return html_pattern.sub(r'', string)

    def del_duplicate_tag(self, string: str):
        spl_string = string.split()
        return ' '.join(sorted(set(spl_string), key=spl_string.index))

