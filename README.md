# Siamese_attribute_extraction
This is a guidance of how to use the proposed Siamese based model for person entity attribute extraction, 
the method performs as well with other entity attibutes.

Last page update: **12/04/2019**

# Prerequiries
1. Download the project.
    
    git clone https://github.com/hycyc/Siamese_attribute_extraction
    
2. Download the [GLoVe embeddings](http://nlp.stanford.edu/data/glove.6B.zip) and copy the embbedings to the glove folder within Siamese_attribute_extraction.
3. Download the [preprocessed wikipedia data](https://www.informatik.tu-darmstadt.de/ukp/research_6/data/lexical_resources/wikipedia_wikidata_relations/), the [person related sentences](https://pan.baidu.com/disk/home?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#/all?vmode=list&path=%2Fsiamese_attribute_extraction%2Fpreprocessed_data211), [the sentence pairs](https://pan.baidu.com/disk/home?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#/all?vmode=list&path=%2Fsiamese_attribute_extraction%2Four_data211) and [the trained models](https://pan.baidu.com/disk/home?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#/all?vmode=list&path=%2Fsiamese_attribute_extraction%2Ftrainedmodels) and copy their folders into the path Siamese_attribute_extraction. You can also download all the mention data as [siamses_attribute_extraction](https://pan.baidu.com/disk/home?errno=0&errmsg=Auth%20Login%20Sucess&&bduss=&ssnerror=0&traceid=#/all?vmode=list&path=%2Fsiamese_attribute_extraction%2Four_data211) from the netdisk with code C4MH.

# Test with the pre-trained models

    python evaluate_on_our_data.py model_Siamese ./preprocessed_data211/train_data_single_sentences.json ./preprocessed_data211/eval_data_single_sentences.json 
    
There will be four files as output, the groundtruth label vector(Siamese_grountruth.mat), the prediction matrix(Siamese_predictions.mat),
the micro p-r curve data(Siamese_curve.dat) and the macro p-r curve data(Siamese_curve-macro.dat). The evaluation accuracy and F1 score will be printed on the screen.

# Re-implement with your data
All data should be pre-processed into a json file, with each element as a sentence containing its sentence tokens, concerned entities with their positions and the KB_ID of the attribute(with only two entities and one attribute for each sentence).
Please modify the original data path, the attributes you concern and the data split proportion in the file and pre-processed the data by:
    
    python all_data_preprocessing.py
    
Group sentences into pairs with the proposed strategy by specifying your data path, the attributes you concern, number of sentence pairs you want and running:
    
    python create_sentence_pairs.py
    
Train the Siamese part as the first step with:
    
    python model_train.py model_Siamese ./preprocessed_data211/train_data_single_sentences.json ./our_data/train_pairs.json ./preprocessed_data211/eval_data_single_sentences.json ./our_data/eval_pairs.json
    
Further train the prediction part as the second setp with:
    
    python model_train_further.py model_Siamese ./preprocessed_data211/train_data_single_sentences.json ./our_data/train_pairs.json ./preprocessed_data211/eval_data_single_sentences.json ./our_data/eval_pairs.json
    
Evaluate the trained model with:
    
    python evaluate_on_our_data.py model_Siamese ./preprocessed_data211/train_data_single_sentences.json ./preprocessed_data211/eval_data_single_sentences.json
    
For the model LSTM-baseline, CNN, Context-sum and Context-weighted, please replace the data as required and remember change the model name.


    
    
    
    
