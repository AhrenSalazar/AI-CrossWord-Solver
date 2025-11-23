# **Crossword Answer Generator (Seq2Seq with Attention)**

This project implements a sequence-to-sequence (Seq2Seq) model with an attention mechanism to generate answers for crossword clues. The model is built using TensorFlow and Keras, and a Gradio interface is provided for interactive demonstration.


The goal of this project is to create an AI model that can accurately predict crossword answers given a clue. Using an encoder-decoder LSTM architecture combined with an additive attention mechanism to handle the variable lengths and complications of natural language processing tasks.


## **Features**

Seq2Seq LSTM Model: Utilizes a Long Short-Term Memory (LSTM) network for sequence processing.

Attention Mechanism: Employs Additive Attention to focus on relevant parts of the clue when generating each character of the answer.

Beam Search Decoding: Enhances prediction accuracy by exploring multiple possible sequences during inference.

Interactive Demo: Integrated with Gradio for an easy web interface.

Persistent Model Storage: Saves best model weights and vectorizer vocabularies for reusability.


## **Data**

The model is trained on a dataset of crossword clues and their corresponding answers. The data is loaded from a CSV file (nytcrosswords.csv).

Pulled from:
https://www.kaggle.com/datasets/darinhawley/new-york-times-crossword-clues-answers-19932021

Clue Preprocessing: Clues are converted to lowercase, punctuation is stripped, and then tokenized into word sequences.

Answer Preprocessing: Answers are converted to uppercase, stripped of whitespace, and then tokenized into character sequences. Special tokens like [SOS] (Start Of Sequence) and [EOS] (End Of Sequence) are added.

## **Key Data Parameters:**

CLUE_SEQ_LEN: Maximum sequence length for clues (e.g., 40 words).

ANS_MAX: Maximum character length for answers (e.g., 12 characters).

ANS_SEQ_LEN: Total answer sequence length including [SOS] and [EOS] (e.g., ANS_MAX + 2).


## **Model Architecture**

Encoder:
An Embedding layer to convert clue word IDs into the dense vectors.
An LSTM layer that processes the embedded clue sequence and outputs its hidden states and context states.

Decoder:
An Embedding layer for answer character IDs.
An LSTM layer initialized with the encoder's final states.

Attention Mechanism:
An AdditiveAttention layer connects the decoder's output at each time step with the encoder's outputs from all time steps, allowing the decoder to selectively 'attend' to parts of the clue.

Output Layer:
A Dense layer with softmax activation predicts the probability distribution over the answer character vocabulary for each time step.

Hyperparameters:
ENC_EMB_DIM: Encoder embedding dimension (e.g., 256).

DEC_EMB_DIM: Decoder embedding dimension (e.g., 256).

ENC_UNITS: Encoder LSTM units (e.g., 256).

DEC_UNITS: Decoder LSTM units (e.g., 256).

BATCH_SIZE: Training batch size (e.g., 128).

EPOCHS: Number of training epochs (e.g., 20, with early stopping).


## **Training!**

The model is trained using sparse_categorical_crossentropy loss and the Adam optimizer. 

Training features:

ModelCheckpoint: Saves the model weights (best_weights.weights.h5) that achieved the lowest validation loss, preventing overfitting.

ReduceLROnPlateau: Dynamically reduces the learning rate when the validation loss stops improving, helping the model converge better.

Overfitting Observation: Training history plots often reveal that the model starts to overfit around 4-5 epochs, where validation loss begins to increase while training loss continues to decrease.

# Inference and Beam Search

For generating answers, a separate inference model is constructed using the encoder_model and decoder_model. This allows for predicting one character at a time.

encoder_model: Takes a clue and outputs the encoder's final states and hidden states.

decoder_model: Takes a single character input, previous decoder states, and encoder outputs (for attention), then predicts the next character and new decoder states.

Beam Search: This decoding strategy explores multiple candidate sequences simultaneously, improving the quality of generated answers by maintaining the beam_width most probable partial sequences at each step.


## **Gradio Demo**

An interactive web demo allowing users to input a crossword clue and receive top-k predicted answers.

Accessing the Demo
Run the Gradio cell, it should provide a link and it can be accessed through that link or in the environment if using colab


## **Setup and Installation**

Clone the Repository:

git clone <your-repository-url>
cd <your-repository-name>
Mount Google Drive (Colab Specific): The project expects data and model artifacts to be stored on Google Drive. If running in Colab, ensure your Google Drive is mounted at /content/drive.

Install Dependencies:

pip install -q gradio scikit-learn tensorflow pandas numpy matplotlib

Prepare Data:

Place your nytcrosswords.csv file in /content/drive/MyDrive/Colab Notebooks/ (or update CSV_PATH in the notebook).
Ensure the directory /content/drive/MyDrive/crossword_model/ exists or is created for saving model artifacts (BASE_DIR).

Usage

Run the Colab Notebook: Execute all cells in sequential order.

Training: The training process will automatically save the best_weights.weights.h5 based on validation loss.

Inference: Once the inference models are built and weights loaded, you can use the beam_search function directly or through the Gradio interface.

Gradio Interface: After running the Gradio cell, open the provided public URL in your browser to interact with the model.

## **Future Improvements**

### More Data and Data Augmentation:

Neural networks live on data so a more diverse training example of clues and answers can help the model learn more robust patterns and handle a wider variety of linguistic catches. Data augmentation (paraphrasing clues, injecting minor noise) can also increase the training set size and make the model more diverse to variations in real-world clues.

### Increase Model Capacity:

Larger embedding dimensions (ENC_EMB_DIM, DEC_EMB_DIM) and more units in the LSTM layers (ENC_UNITS, DEC_UNITS) allow the model to capture more complicated relationships and finer details in the input sequences. This can lead to a deeper understanding of semantic connections between clues and answers.

### Deeper LSTMs or Bidirectional Encoder:

Stacking multiple LSTM layers (making them 'deeper') enables the model to learn hierarchical features. A bidirectional encoder allows the model to process the clue in both forward and backward directions, providing a richer contextual understanding of each word in the clue.

### Explore Transformer Architectures:

Transformer models (like BERT, GPT, T5) have become the state-of-the-art for sequence-to-sequence tasks due to their self-attention mechanisms, which can capture long-range dependencies more effectively than LSTMs. While a huge and complicated change, this would improve performance potential.

### Hyperparameter Tuning & Regularization:

Finely tuning hyperparameters (learning rate, batch size, optimizer settings) can help the model converge more efficiently and to a better optimum. Techniques like Dropout or L1/L2 regularization can prevent overfitting, especially when increasing model capacity, ensuring the model generalizes well to unseen data.

### Advanced Decoding Strategies:

While beam search is already employed, further experimentation with higher beam_width values (balancing computational cost with potential gain) or more sophisticated search algorithms could yield better results by exploring a larger space of possible answers.

### Error Analysis and Targeted Improvements:

By systematically analyzing the types of errors the model makes (semantic vs. syntactic errors, handling of idioms, specific categories of clues), you can identify weaknesses and design targeted improvements, such as adding specific features or refining data pre-processing for those problematic areas.


## **Acknowledgements**

TensorFlow and Keras for the deep learning framework.
Gradio for the interactive web interface.
scikit-learn for utility functions.
