#!/usr/bin/env python
# coding: utf-8





def main():
    
    import json
    import datasets
    import argparse
    import spacy

    spacy.prefer_gpu()
    from tqdm import tqdm

    import spacy.cli
    



    parser = argparse.ArgumentParser(description='--- NER scorer ----')
    parser.add_argument("--dataset", type=str,default='dataset/train-full.json', help='sentence pairs')
    parser.add_argument("--output_filename", type=str, default='train_ner_expanded.json', help='save dataset name')
    parser.add_argument("--ner_model", type=str, default='en_core_web_md',help="NER model name")

    
    hp, _ = parser.parse_known_args()
    

    nlp = spacy.load(hp.ner_model) # load model

    def calculate_information_density(text1, text2):
    
        # Process text1
        doc1 = nlp(text1)
        entities1 = [(ent.text, ent.label_) for ent in doc1.ents]
        information_density1 = len(entities1) / len(doc1)
    
        # Process text2
        doc2 = nlp(text2)
        entities2 = [(ent.text, ent.label_) for ent in doc2.ents]
        information_density2 = len(entities2) / len(doc2)
    
        # Calculate the average information density
        average_information_density = (information_density1 + information_density2) * 0.5
    
        return average_information_density
   

    with open(hp.dataset, 'r') as f:
        data = json.load(f)
        
    for i, date_tuple in tqdm(enumerate(data), total = len(data)):
        try:
            left = data[i]['input'].split('   ')[0]
            right = data[i]['input'].split('   ')[1]
        except IndexError:
            left = data[i]['input'].split('product 2:')[0]
            right = "product 2:" + data[i]['input'].split('product 2:')[1]   
    
      
        ner_score_ = calculate_information_density(left, right)  
  
        data[i]['ner_score'] = ner_score_
    
    
    with open(hp.output_filename, 'w') as f:
        json.dump(data, f, indent=2)
    
    
        
if __name__ == "__main__":
    main()



