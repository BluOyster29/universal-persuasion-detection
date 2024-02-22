import numpy as np
import yaml
import json
import argparse
import ast

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

p = argparse.ArgumentParser()
p.add_argument('config_path', type=str, help='Path to the yaml file containing config variables')


def load_config(path: str) -> dict:
    
    """
    args:
        path : path to the config file which is in yaml format
    
    returns:
        dictionary formmated config

    """

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def convert_top_level(tag : str, top_level_only=None) -> str:

    """
    Cheating script that converts full tag to 
    top level only tag, taking into account 
    bugs and mistakes
    
    args:
        tag: a string of the full tag 
        that was annotated by the user 

    returns:
        string:
    """

    if tag == '8-NO-PERSUASION':
        return tag

    elif tag == '8-NO':
        return '8-NO-PERSUASION'

    elif tag == '7-RESSURE-OTHER':
        return '7-PRESSURE'
        
    if top_level_only:
        return '-'.join(tag.split('-')[:2])
    else:
        return tag


def open_training_data(json_path : str, conf_thresh: float=None, 
                        include_persuadee: bool=None):
    
    """
    Script opens the training data that is a josnl file. The training 
    data file has been processed from previosu scripts that are not 
    available in this repo

    args:
        json_path : path to the json 

    kwargs:
        conf_thresh : float that indicates what level of confidence the 
                      scrip will accept. For the training data there is 
                      only 2 annotators so the confidence can only be 1 
                      or 0.5 
                      Testing data has up to 9 annotators so can be any-
                      where between 0 and 1

    returns:
        training_data: a list of strings that are the annotations from the 
                       dataset.
    """

    training_data = []
    heldback = []
    
    with open(json_path) as file:
        
        for line in file.readlines():
            line = json.loads(line)
            
            if include_persuadee == True and line.get('persuadee'):
                training_data.append(line)
                
            elif conf_thresh:
                
                if line.get('confidence') and line.get('confidence') > conf_thresh:
                    training_data.append(line)
                    
                else:
                    if not line.get('persuadee'):
                        heldback.append(line)
                    
            elif not line.get('persuadee'):
                training_data.append(line)
                
    return training_data


def process_data_2(tag2idx, training_data):

    train_x = []
    train_y = []

    for index in range(len(training_data)):
        
        x = training_data[index]['text']
        y = training_data[index]['gold_tag']

        splat_tag = y.split('/')

        if len(splat_tag) > 1:

            for tag in splat_tag:

                train_y.append(tag2idx[tag])
                train_x.append(x)


        else:
            train_x.append(x)
            train_y.append(tag2idx[y])

    return train_x, train_y 


def vectorise_tags(config, training_data: list, testing_data: list, 
            ammount_suppliment: int=None):

    """
    The script vectorizers the tags which are strings 
    into multi-hot encoded vectors to represent multilabel
    annotations

    args: 
        training_data : dictionary containing each annotation
                        and its meta data 
        testing_data  : same format as training_data
    
    kwargs:
        ammount_suppliment : integer describing how much data 
                             to take from the training set and put 
                             into the testing set. Necessary if not 
                             all labels are represented in training 

    returns: 
        train_x : list of training x inputs  
        train_y : list of training labels
        test_x : list of testing x inputs 
        test_y : list of testing labels
    """


    train_x = [i['text'] for i in training_data]
    

    # Adding some examples to testing set so every label is represented in the testing set 
    test_x = [i['text'] for i in testing_data]

    if config.get('model_config').get('onevsrest'):
        train_y = [np.clip(i['multilabel'],0,1) for i in training_data]
        test_y = [np.clip(i['multilabel'],0,1) for i in testing_data]
    else:
        tag2idx = config.get('taglist')

        train_x, train_y = process_data_2(tag2idx, training_data)
        test_y = [tag2idx[i['gold_tag']] for i in testing_data]

    if ammount_suppliment:
        test_x += train_x[-ammount_suppliment:]
        test_y += train_y[-ammount_suppliment:]
        
        train_x = train_x[:-ammount_suppliment]
        train_y = train_y[:-ammount_suppliment]
    
    print(f'Total Training: {len(train_x)}')
    print(f'Total Testing: {len(test_x)}')

    config['num_training'] = len(train_x)
    config['num_testing'] = len(test_y)

    return config, train_x, train_y, test_x, test_y


def tag_to_vec(tag: str, tag2idx: dict, top_level_only: bool=None) -> str:
    
    """
    Rather messy script that converts a string label 
    into the vectorised version. E.g 1-RAPPORT -- [1,0,..n]

    args: 
        tag : the tag as a string
        tag2idx : dictionary mapping tags to index

    kwargs:
        top_level_only : boolean whether or not to use top level
                         tags

    returns:
        empty_vec : multihot label 
    """

    empty_vec = np.zeros(len(tag2idx.keys()))
    
    if tag == 'NO':
        print('fuk')
    if len(tag.split('/')) > 1:
        multitag = tag.split('/')
        
        if top_level_only:
            tag = convert_top_level(tag, top_level_only=True)

        for tag in multitag:
            
            if tag == '8-NO':
                empty_vec[tag2idx['8-NO-PERSUASION']] = 1
        
            elif tag == '7-RESSURE-OTHER':
                empty_vec[tag2idx['7-PRESSURE-OTHER']] = 1
            
            elif tag == '7-RESSURE':
                empty_vec[tag2idx['7-PRESSURE']]

            else:
                empty_vec[tag2idx[tag]] = 1
            
    else:
        if top_level_only:
            tag = convert_top_level(tag, top_level_only=True)
            
        if tag == '8-NO':
            empty_vec[tag2idx['8-NO-PERSUASION']] = 1
        
        elif tag == '7-RESSURE-OTHER':
            empty_vec[tag2idx['7-PRESSURE-OTHER']] = 1
        elif tag == '7-RESSURE':
            empty_vec[tag2idx['7-PRESSURE']]
        else:
            empty_vec[tag2idx[tag]] = 1
        
    return empty_vec


def chooze_vectorizer(vec_type: int ,stop_words: str=None, ngram_range: tuple=None):
    
    """
    SCript takes the config for the vectorizer and returns 
    the sklearn vectorizer to be used for vectorizing the x inputs
    as well as a placeholder name for the pipeline 

    args:
        vec_type  : 1 or 2, one being count vectorizer and 2 being tfidf
        lowercase : whether or not to convert text to lowercase before 
                    building featuer map 
        stop_words : if blank no stopwords are removed otherwise a string
                     with the language of stopwords
        ngram_range: tuple representing the ngrams for example (1, 1) which
                    is the default will convert the tokens into unigrams. (2,2)
                    will use only bigrams (1,2) will use both unigram and bigrams

    returns:
        vec_name : placeholder name for sklearn's pipeline script
        vectorizer : sklearn's vectorizer object
    """

    if not ngram_range:
        # default ngram range if no ngram chosen
        ngram_range = (1,1)
        
    # count vectorizer uses word frequencies to populate vectors
    if vec_type == 1:
        vec_name = 'count'
        vectorizer = CountVectorizer(lowercase=True, ngram_range=ngram_range,
                                    stop_words=stop_words)
        
    # tfidf vectorizers uses tfidf representation
    elif vec_type == 2:
        vectorizer = TfidfVectorizer(lowercase=True, ngram_range=ngram_range,
                                    stop_words=stop_words)
        vec_name = 'tfidf'
        
    return vec_name, vectorizer


def get_vectorizer(config : dict):

    """
    Script for returning the vectorizer, I am not sure
    why i haven't combined the above with this one
    """
    
    vectorizer_config = config.get('vectoriser_config')

    vectorizer_name, vectorizer = chooze_vectorizer(vec_type=vectorizer_config.get('type'), 
                                        ngram_range=ast.literal_eval(vectorizer_config.get('ngram_range')),
                                        stop_words=vectorizer_config.get('stop_words'))

    return vectorizer_name, vectorizer

def preprocess(config):


    """
    Main script for preprocessing, takes the config file 
    then returns the vectorizer and training data for the 
    model

    args:
        config : dictionary containing the processed configuration
                 that the user generated in the yaml file

    returns:
        vectorizer_name : placeholder to describe vectorizer
        vectorizer : sklearn vectorizer objcet 
        train/test_x : list of strings 
        train/test_y : list of multihot vectors 
    """

    training_data = open_training_data(config.get('training_data').get('path'), 
        conf_thresh=config.get('training_data').get('conf_threshold'))

    testing_data = open_training_data(config.get('testing_data').get('path'), 
        conf_thresh=config.get('testing_data').get('conf_threshold'))

    config, train_x, train_y, test_x, test_y = vectorise_tags(config, training_data, testing_data, 
        ammount_suppliment=config.get('suppliment_testing'))

    vectorizer_name, vectorizer = get_vectorizer(config)

    return config, vectorizer_name, vectorizer, train_x, train_y, test_x, test_y

if __name__ == '__main__':
    args = p.parse_args()
    config, vectorizer_name, vectorizer, train_x, train_y, test_x, test_y = preprocess(args.config_path)
