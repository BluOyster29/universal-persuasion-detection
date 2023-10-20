import argparse
import yaml
import pickle 
import ast 
from datetime import datetime
import os 

from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report

from preprocess import preprocess

mlb = MultiLabelBinarizer()


"""

The main training script for training a model to detect
persuasive strategies in text. The training data consists
of annotations from transcripts with persuasion strategies
as labels. The script processes the texts and converts them
into either count vectors or tfidf processed vectors as well 
as converting labels into multihot vectors. The script then 
trains a model using the user's chosen configuration and outputs
the model to a the models/ folder.

"""


p = argparse.ArgumentParser()
p.add_argument('config_path', type=str, help='Path to the yaml file containing config variables')


def load_config(path: str) -> dict:
    
    """
    Loads the yaml formatted config as a 
    dictionary
    """

    with open(path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config


def chooze_model(model_config):
    
    """
    From the model config a model architecture 
    is chosen, either a multilayer perceptron or 
    support vector machine. User can also choose 
    whether or not the model is wrapped ina onevsrest
    class

    returns:
        model: the placeholder name for the model
        clf : classifier object
    """


    if model_config.get('type') == 1:
        model = 'mlp'
        clf = MLPClassifier(max_iter=model_config.get('max_iter'), verbose=True)
        
    elif model_config.get('type') == 2:
        clf = SVC(verbose=True)
        model = 'svm'

    if model_config.get('onevsrest'):
        model = 'mlp_1vrest'
        return model, OneVsRestClassifier(clf, n_jobs=-1, verbose=True)
    
    else:
        return model, clf


def train_classifier(config, vectorizer_name= None,vectorizer=None,
                            train_x=None, train_y=None, test_x=None, test_y=None):

    """
    Top level train script for training the mdoel 

    args:
        config: path to config file 

    kwargs:
        vectorizer_name: placeholder name for model
        vectorizer: sklearn classifier 
        train_x/yetc. : preprocessed training/testing data

    returns:
        clf_pipe: pipeline for classifying data 
        test_x,test_y: list of testing examples for evaluation
    """


    if not vectorizer_name:
        config, vectorizer_name, vectorizer, train_x, train_y, test_x, test_y = preprocess(config)
    model_config = config.get('model_config')

    model, clf = chooze_model(model_config)
    
    clf_pipe = Pipeline([
                     (vectorizer_name, vectorizer),
                    (model, clf),
                 ])

    #train_y = mlb.fit_transform(train_y) for mlp not onevs rest
    clf_pipe.fit(train_x, train_y)

    return clf_pipe, test_x, test_y 


def decode_tup(tup):
    
    """
    Short script for translating the ngram_range
    tuple into string to be appended to the model
    output path
    """

    tup = ast.literal_eval(tup)
    
    if tup == (1,1):
        tup = 'unigrams'
        
    elif tup == (1,2):
        tup = 'uni_bigrams'
        
    elif tup == (2,2):
        tup = 'bigr_bigr'
        
    elif tup == (2,3):
        tup = 'bigr_trigr'
        
    elif tup == (3,3):
        tup = 'tri_tri'
            
    return str(tup)


def output_eval(trained_model, config):
    
    """
    Script for outputting the model and config file.
    More information is added to the config file during 
    training, this can be accessed after exporting
    """

    path = config_to_path(config)
    with open(path, 'wb') as file:
        print(f'Outputting model to {path}')
        pickle.dump(trained_model, file)

    config_path = f'{path[:-4]}.yaml'
    with open(config_path, 'w+') as file:
        yaml.dump(config, file)
        print(f'Outputting config to {config_path}')

def evaluate(trained_model, test_x, test_y, config, verbose=None):

    idx2tag = {idx : tag for tag, idx in config.get('taglist').items()}
    
    try:
        labels = list(sorted(set([idx2tag[i] for i in test_y])))
    except:
        labels = list(idx2tag.values())
    
    
    preds = trained_model.predict(test_x)
    report = classification_report(preds, test_y, target_names = labels,zero_division=True, output_dict=True)
    
    text = classification_report(preds, test_y, target_names = labels,zero_division=True)
    print(text)
        
    config['classification_report'] = report
    config['time'] = datetime.now().strftime('%d-%m-%Y %H:%M')

    if config['model_config'].get('output_path'):
        output_eval(trained_model, config)
    
    return config, text

def config_to_path(config):
    
    model_config = config.get('model_config')
    vectoriser_config = config.get('vectoriser_config')
    folder_path = model_config.get("output_path")
    folder = [f'{folder_path}model_{int(len(os.listdir(folder_path))/2)}']
    
    for key, val in vectoriser_config.items():
        
        if key == 'type':
            
            if val == 1:
                folder.append('bow')
                
            elif val == 2:
                folder.append('tfidf')
        
        if key == 'ngram_range':
            folder.append(decode_tup(val))

    for key, val in model_config.items():
        
        if key == 'type':
            
            if val == 1:
                folder.append('mlp')
            
            elif val == 2:
                folder.append('svm')
                
        if key == 'onevsrest':
            
            folder.append(str(val))
    
    path = ['_'.join(folder)]
    path.append('.pkl')
    path = ''.join(path)
    return path

def batch_train(list_of_configs, verbose=None):
    
    num = 1
    
    print('---' * 10)
    print('\n\n')
    print(f'Training Model 1...')
    for config in list_of_configs:
        
        vectorizer_name, vectorizer, train_x, train_y, test_x, test_y = preprocess(config)
        clf, test_x, test_y = train_classifier(config, vectorizer_name=vectorizer_name,vectorizer=vectorizer,
                            train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
        evaluate(clf, test_x, test_y, config, verbose=verbose)
    
    print('\n\n')
    print('---' * 10)
    print('Finished!')


def batch_train(list_of_configs, verbose=None):
    
    num = 1
    
    print('---' * 10)
    print('\n\n')
    print(f'Training Model 1...')
    for config in list_of_configs:
        
        vectorizer_name, vectorizer, train_x, train_y, test_x, test_y = preprocess(config)
        clf, test_x, test_y = train_classifier(config, vectorizer_name=vectorizer_name,vectorizer=vectorizer,
                            train_x=train_x, train_y=train_y, test_x=test_x, test_y=test_y)
        evaluate(clf, test_x, test_y, config, verbose=verbose)
    
    print('\n\n')
    print('---' * 10)
    print('Finished!')


if __name__ == '__main__':
    args=p.parse_args()
    config = load_config(args.config_path)
    trained_model, test_x, test_y = train_classifier(config)
    evaluate(trained_model, test_x, test_y, config)



"""
int (1 or 2 so far) refers to which type of model 
                   architecture to use, in this case it is either the 
                   multilayer perceptron or support vector machine """