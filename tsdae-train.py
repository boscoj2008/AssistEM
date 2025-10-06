#!/usr/bin/env python
# coding: utf-8

# code adapted from: https://sbert.net/examples/sentence_transformer/unsupervised_learning/TSDAE/README.html

# to run use: python tsdae-train.py --dataset dataset/train-full.json --batch_size 32 --epochs 10



def main():
    
    import json
    import datasets
    import argparse

    from sentence_transformers import SentenceTransformer, LoggingHandler
    from sentence_transformers import models, util, evaluation, losses
    from torch.utils.data import DataLoader
    from sentence_transformers.datasets import DenoisingAutoEncoderDataset




    parser = argparse.ArgumentParser(description='--- TSDAE ----')
    parser.add_argument("--dataset", type=str,default='dataset/train.json', help='sentence pairs')
    parser.add_argument("--output", type=str, default='output/tsdae-model', help='save model folder')
    parser.add_argument("--model_name", type=str, default='bert-base-uncased',help="model_path or name")
    parser.add_argument("--batch_size", type=int, default=8, help='batch size')
    parser.add_argument("--epochs", type=int, default=10, help='epochs')
    
    hp, _ = parser.parse_known_args()


    # Vector model
    word_embedding_model = models.Transformer(hp.model_name)
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), "cls")
    model = SentenceTransformer(modules=[word_embedding_model, pooling_model])


    # Compile dataset 
    with open(hp.dataset, 'r') as f:
        data = json.load(f)
        train_sentences=list()
        for item in data:
            train_sentences.append(item['input']) 
        # final dataset    
        dataset = datasets.Dataset.from_dict({'text': train_sentences})
    
    
    # Create the special denoising dataset that adds noise on-the-fly
    train_dataset = DenoisingAutoEncoderDataset(dataset['text'])
    
    # DataLoader to batch your data
    train_dataloader = DataLoader(train_dataset, batch_size=hp.batch_size, shuffle=True)

    # Use the denoising auto-encoder loss
    train_loss = losses.DenoisingAutoEncoderLoss(
        model, decoder_name_or_path=hp.model_name, tie_encoder_decoder=True
    )

    # Calibrates encoder
    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=hp.epochs,
        weight_decay=0,
        scheduler="constantlr",
        optimizer_params={"lr": 3e-5}, # use default lr
        show_progress_bar=True,
    )



    model.save(hp.output)
    

if __name__ == "__main__":
    main()



