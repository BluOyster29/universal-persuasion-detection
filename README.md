# Home

The following folder contains code and data from the universal persuasion detection projhect

# Folder Directory

`code`: Folder containing all code used for the following -

    - Process raw data from p4good into a format for Doccano
    - Extract and preprocess the exported data from Doccano
    - Calculate Agreement metrics
    - Post process the extracted data from Doccano
    - Output newly formatted data in for publication
    - Train/evaluate models on the processed data

`data`: Multiple folders containing all textual data

`papers`: Resources used in papers/presentations

# Installation

1. Clone the repository  `<br>`
   `git clone <path_to_repo>`
2. Create a virtual environment `<br>`
   `venv persuasion_strategies_env`
3. Install requirements from requirements.txt `<br>`
   `pip install -r './requirements.txt'`
4. Fill in Configuration file
   - See below
5. Run train.p `<br>`
   `python ./code/train.py 'config.yaml'`

# Config File for Sklearn Classifiers

Find in the root folder a file named config.yaml. With this file you can change parameters
for training/data processing.

The file looks a bit like this

```
suppliment_testing: 0 --- Indicates the number of training examples to put in the testing set 
                          before training, only necessary if not all labels are present in testing 

taglist:

  1-RAPPORT: 0
  2-NEGOTIATE: 1
  3-EMOTION: 2
  4-LOGIC: 3
  5-AUTHORITY: 4
  6-SOCIAL: 5
  7-PRESSURE: 6
  8-NO-PERSUASION: 7
  NO-TAG: 8

testing_data:
  conf_threshold: 0.6 --- float between 0 and 1, the confidence score represents the proportion
                            of annotators that chose the majority tag, for example 0.5 out of two 
                            annotators indicates %50 of annotators chose that tag, a higher
                            confidence implies a greater number of annotators choosing a tag

  output_path: data/dataloaders/ --- not currently in use
  path: data/post_processed/testing_data.jsonl --- path to data

training_data:
  conf_threshold: 0.5
  output_path: data/dataloaders/
  path: data/post_processed/training_data.jsonl

vectoriser_config:
  type: 1 --- word count vectorizer=1, tfidf vectorizer=2
  lowercase: true --- Have the vectorizer process to tokens to lowercase
  ngram_range: (2,2) --- Choose the ngram range (1,1) for only unigrams, 
                          (1,2) for unigrams and bigrams, (2,2) for only bigrams etc.
  stop_words: english --- Have the vectorizer remove english stop words

model_config:
  type: 1 --- Support Vector Machine=1, Multilayer Perceptron=2
  onevsrest: true --- Wrap the model in sklearn's onevsrest, this will train n amount 
                        of models based on number of labels
  output_path : models/ --- Folder for outputting the model, the script will provide 
                            a name for the model based on the chosen config

```

Once you have chosen your config you can run the train.py script that is in the code folder with the following command.

The script should run, once the model is trained the script will output a copy of your config and the model to `models` folder.

# BERT Classifier

The notebook `BERT_Classifier.ipynb` is best run on a device that has a GPU for fine tuning the BERT model. If you open the notebook in colab there are a few things you need to make sure to do to make the notebook work.

1. Drag the training/testing data in to the session

- You need to drag and drop `data/training.csv` and `data/testing.csv` into the tab that has the notebook open. You need to make sure the `files` tab is open. If done correctly you should see the files at the following path `/content/training.csv` and `/content/testing.csv`

2. Drag `BERT_Module.py` into the session

- Follow the above but this time you need to find `BERT_Module.py` which you can find in `code/BERT_Module.py`
- Drag and drop into the session and you should see it pop up on the sidebar
- Your session should look like this

<image src="papers\figs\colab_screenshot.png" style=" text-align=center;">

3. Run the notebook

If all has worked the notebook should run fine. The training script will output into a folder named Tensorboard, I will add a further tutorial later on how to use this folder
