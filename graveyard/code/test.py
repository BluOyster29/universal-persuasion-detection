import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

def preprocess_text(text):
    
    """
    Preprocesses the given text by converting it to lowercase, removing numbers and punctuation,
    tokenizing the text, removing stopwords, lemmatizing words, and joining tokens back into a single string.
    
    Args:
        text (str): The text to be preprocessed.
    
    Returns:
        str: The preprocessed text.
    """
    
    # Convert text to lowercase
    text = text.lower()
    
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    
    # Tokenize text
    tokens = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    
    # Lemmatize words
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # Join tokens back into a single string
    preprocessed_text = ' '.join(tokens)
    
    return preprocessed_text

# Example usage

if __name__ == "__main__":
    inputtext = "This is an example sentence. It needs to be preprocessed!"
    outputtext = preprocess_text(inputtext)
    print(outputtext)
