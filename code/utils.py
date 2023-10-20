import pickle

class exportArgs:
    
    def __init__(self, json_path, exclude_users, remove_persuadee,
                        max_transcripts, tagset, fix_newlines, use_true_names,
                         pickle_file, dataset_version=None, top_level_only=None):
        
        self.json_path         = json_path
        self.exclude_users     = exclude_users.split(' ')
        self.remove_persuadee  = remove_persuadee
        self.max_transcripts   = max_transcripts
        self.tagset            = tagset
        self.fix_newlines      = fix_newlines
        self.use_true_names    = use_true_names
        self.pickle_file       = pickle_file
        self.dataset_version   = dataset_version
        self.top_level_only    = top_level_only

def import_dataset(path):

    with open(path, 'rb') as file:
        dataset = pickle.load(file, encoding='bytes')

    print(f'Loaded dataset from {path}')
    print(dataset)

    return dataset