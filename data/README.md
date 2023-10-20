# Data Folder 

`exported_from_doccano`:
- 5-transcripts: initial 5 transcripts annotated by the project group and annotators to analyse agreement 
- 20-transcripts: larger training set to further analyse agreement and fine tune anotation instructions
- dialogue_act_final: 5 transcripts annotated using dialogue acts, completed by project group.
- main_markup: contains final markup completed by annotator pairs, each annotator is part of 4 batches of ~50 transcripts

`P4GoodMain`: 
- Raw data from P4Good paper as hosted from their website 
- Data is used to be processed to go on Doccano 

`post_processed`: 
- heldback.json: transcripts annotated with utterances that are disagreed upon. Possible use for inviting other annotators to choose from the disagreed tags, breaking the deadlock. 
- testing_data.jsonl: testing set, data is procesesd from the main markup and formatted to be easily publishable
- training_data.jsonl: same format as testing_data.jsonl but much more transcripts

`raw_data`: 
- final_data_for_markup: this is processed data from <i>rap4goodmain</i>, procesed to go on doccano with each user pair segregated
- 100_transcripts_da.jsonl: 100 transcripts to be annotated for dialogue acts, already on Doccano, currently being annotated