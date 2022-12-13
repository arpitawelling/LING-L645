## Project for CSCI B-659 and Independent Study CSCI-Y 790

### Predictive Text for Chukchi

#### Highlights:

This project focuses on building a text prediction for under-resourced and indigenous language: Chukchi. The focus of this project was to build on an existing project that develops a unigram-bigram model on the same language. 

#### Methods:

1. Data Preprocessing:

This model read the given lines of data and then created a corpus according to the sequence length. For example, for sequence length 10, if the first row of the data set is йъйыӄык ныӄэԓпэратӄэн вытэчгытрыӄэргыԓьын йыӈэттэт and second row is мыкыӈ нывытрэтӄин чеԓгатвытрыԓьо ынӄорыым вытэчгытрыԓьо , the data for neural network would look like this:
     X        |     Y      |
| ----------- | -----------|
| йъйыӄык ны  | ӄ          |
| ъйыӄык ныӄ  | э          |

where X is the input to neural network and Y is the expected output.

This project explores 2 different Neural network architectures. After the data is processed from the .tsv file, the 2 different models handle it in different ways. 


2. Details on Model 1

The first model that I tried was a basic LSTM model with 2 layers of LSTM and a Dense layer at the end. After creating the corpus as explained in Data Preprocessing step, we have shape of X as (len(sentences),SEQUENCE_LENGTH,len(characters)) and shape of Y as ((len(sentences),len(characters)) where len(sentences) are the total number of sentences created in our corpus and len(characters) are the total number of different characters in the whole corpus. 

This model had the following hyperparameters:
Learning rate: 0.0001
Batch size: 128
Epochs: 100

Although this method was pretty straight forward, it did not predict words at all. Although it was predicting characters, the accuracy was poor on validation set.

3. Details on Model 2

For this model, I added an embedding layer, one LSTM layer with dropout and a Dense layer at the end. After the data preprocessing step, we have shape of X as (len(sentences), SEQUENCE_LENGTH, 1) and Y has a shape of (len(sentences), len(characters)).

This model has the following hyperparamteres:
Learning rate: 0.0001
Batch size: 128
Epochs: 70
Dropout: 0.5

This method was able to build words and predict some words when given most of the context.

#### Evaluation and Results:

For validating the model, I am giving it a sentence just like we would do in a keyboard.
For example, if I want to type ӄԓявыԓя риӄукэтэ ивнин ытри ынкы варкыт гынин ӈэвъэн гэԓгыԓин мэмыԓя рагтыгъэ , then I would first type ӄ and then the model would predict 3 characters and using these 3 characters, it will build 3 different words. This can be extended to any number of words. I choose 3 as that is generally what we see on our smart phone keyboards. 
The output would look something like this:

Actual sentence: ӄԓявыԓя риӄукэтэ ивнин ытри ынкы варкыт гынин ӈэвъэн гэԓгыԓин мэмыԓя рагтыгъэ   
Clicks: 'ԓ', 'я', 'в', 'ы', 'ԓ', 'я', ' ', 'и', 'ӄ', 'укэтэ', ' ', 'внин', ' ', 'т', 'р', 'и', ' ', 'н', 'к', 'ы', ' ', 'а', 'р', 'к', 'ы', 'т', ' ', 'ы', 'н', 'и', 'н', ' ', 'эвъэн', ' ', 'э', 'ԓ', 'г', 'ы', 'ԓ', 'ин', ' ', 'э', 'м', 'ы', 'ԓ', 'я', ' ', 'а', 'гтыгъэ', ' '

This output can be interpreted in this way:

When the user inputs ӄ, the model is not able to predict the word so the user would type ԓ. The next input for model is ӄԓ, and the model is still not able to predict the word, so the user now types я. Now the input to model is ӄԓя, the model is still not able to predict the word, so the user would type в. This will go on. The model takes the last 10 characters, as that is the SEQUENCE_LENGTH for the model, to predict the next character.

So if the input is риӄукэтэ и, the model predicted внин which completes the word ивнин.

For evaluating this model, we will count clicks/tokens. For example, for the word ӄԓявыԓя, there are 7 clicks by user and for риӄукэтэ word there are 4 clicks by user so clicks per token will be 11/2 = 5.5

For the test data the clicks/token is 7.13 and for validation set it is 5.74.

#### File structure:

Model 1: Text_prediction-Sequence length 15 new validation.ipynb

Model 2: LSTM_embedding_layer_dropout50_new-seqlength10-epochs70.ipynb

Model 2 prediction on dev and test data: Prediction-dev.ipynb, Prediction-test.ipynb 

Output: Output folder 

Model : LSTM_embeddinglayer_dropout50_epochs70_final.dat

#### References:

Original repository: https://github.com/ftyers/global-classroom/tree/main/chukchi

Text Prediction articles: https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218

https://towardsdatascience.com/word-and-character-based-lstms-12eb65f779c2

Articles on text entry: https://www.yorku.ca/mack/hcimobile02.html

Survey paper: https://www.sciencedirect.com/science/article/pii/S1319157820303360

