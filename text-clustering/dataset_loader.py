import ast



def dataset_load(key):

    key_to_data_path = {
        "Tweet": "data/tweet.txt",
        'Bio': "data/biomedical.txt",
        'SO': "data/stackoverflow.txt",
        'SS': "data/searchsnippets.txt",
        'AG': "data/agnews.txt",
        'G-TS': "data/ts.txt",
        'G-T': "data/t.txt",
        'G-S': "data/s.txt"
    }
    
    key_to_label_path = {
        'Bio': "data/biomedical_label.txt",
        'SO': "data/stackoverflow_label.txt",
        'SS': "data/searchsnippets_label.txt"
    }
    
    data_path = key_to_data_path[key]
    
    if key in key_to_label_path:
        label_path = key_to_label_path[key]
        

    if key == "Tweet":
        with open(data_path) as f:
            l_strip = [s.strip() for s in f.readlines()]
        sentences = [ast.literal_eval(d)["text"] for d in l_strip]
        labels = [ast.literal_eval(d)["cluster"] for d in l_strip]
        
    elif key in ["Bio", 'SO', 'SS']:
        with open(data_path) as f:
            sentences = [s.strip() for s in f.readlines()]
        with open(label_path) as f:
            labels = [int(s.strip()) for s in f.readlines()]
            
    elif key in ['AG', 'G-TS', 'G-T', 'G-S']:
        with open(data_path) as f:
            l_strip = [s.strip() for s in f.readlines()]
        sentences = [d.split("\t")[1] for d in l_strip]
        labels = [int(d.split("\t")[0]) for d in l_strip]
        

    return sentences, labels