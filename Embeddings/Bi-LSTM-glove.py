import pandas as pd
import numpy as np
import tensorflow as tf
import re
import os
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Suppress TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.get_logger().setLevel('ERROR')

def clean_text(text):
    """Clean and preprocess text data"""
    if pd.isna(text):  # Handle NaN values
        return ""
    text = str(text)  # Convert to string
    text = text.lower()
    text = re.sub(r'http\S+', '', text)  # Remove URLs
    text = re.sub(r'@\w+', '', text)     # Remove mentions
    text = re.sub(r'#', '', text)        # Remove hashtags
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.strip()

def load_data(file_path):
    """Load and preprocess dataset"""
    df = pd.read_csv(file_path)
    df = df[['tweet', 'sarcastic', 'sarcasm', 'irony', 'satire', 
            'understatement', 'overstatement', 'rhetorical_question']]
    
    # Handle missing values
    df['tweet'] = df['tweet'].fillna('')
    
    # Clean text
    df['clean_tweet'] = df['tweet'].apply(clean_text)
    return df

def load_glove(filepath):
    """Load GloVe embeddings"""
    embeddings_index = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def plot_correlation_heatmap(df):
    """Plot correlation matrix of sarcasm types"""
    plt.figure(figsize=(10, 8))
    corr = df[['sarcastic', 'sarcasm', 'irony', 'satire', 
              'understatement', 'overstatement', 'rhetorical_question']].corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Sarcasm Type Correlation Matrix')
    
    plot_dir = "plots"
    os.makedirs(plot_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{plot_dir}/sarcasm_correlation_{timestamp}.png", dpi=300, bbox_inches='tight')
    plt.close()

# Parameters
MAX_LENGTH = 50
GLOVE_DIM = 100
BATCH_SIZE = 256
EPOCHS = 20

# Load and preprocess data
df = load_data('../../train/train.En.csv')

# Plot correlations
plot_correlation_heatmap(df)

# Prepare data
X = df['clean_tweet']
y = df['sarcastic']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Tokenization
tokenizer = Tokenizer(oov_token='<OOV>')
tokenizer.fit_on_texts(X_train)
VOCAB_SIZE = len(tokenizer.word_index) + 1

# Load embeddings
glove_embeddings = load_glove('../../glove.6B.100d.txt')

# Create embedding matrix
embedding_matrix = np.zeros((VOCAB_SIZE, GLOVE_DIM))
for word, i in tokenizer.word_index.items():
    if i < VOCAB_SIZE:
        embedding_vector = glove_embeddings.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

# Sequence padding
train_sequences = tokenizer.texts_to_sequences(X_train)
test_sequences = tokenizer.texts_to_sequences(X_test)
train_padded = pad_sequences(train_sequences, maxlen=MAX_LENGTH, padding='post')
test_padded = pad_sequences(test_sequences, maxlen=MAX_LENGTH, padding='post')

# Bi-LSTM Model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(
        input_dim=VOCAB_SIZE,
        output_dim=GLOVE_DIM,
        weights=[embedding_matrix],
        input_length=MAX_LENGTH,
        trainable=False
    ),
    tf.keras.layers.SpatialDropout1D(0.3),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        64, 
        return_sequences=True,
        recurrent_dropout=0.2,
        kernel_regularizer=tf.keras.regularizers.l2(0.001)
    )),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(
        32,
        recurrent_dropout=0.2,
        kernel_regularizer=tf.keras.regularizers.l2(0.001))
    ),
    tf.keras.layers.Dense(64, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
    tf.keras.layers.Dropout(0.4),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(
    loss='binary_crossentropy',
    optimizer=tf.keras.optimizers.Adam(0.0005),
    metrics=['accuracy']
)

# Callbacks
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    min_delta=0.001,
    restore_best_weights=True
)

lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=2
)

# Training
history = model.fit(
    train_padded,
    y_train,
    epochs=EPOCHS,
    validation_data=(test_padded, y_test),
    batch_size=BATCH_SIZE,
    callbacks=[early_stop, lr_scheduler]
)

# Plot training results
def plot_training(history):
    plt.figure(figsize=(14, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train')
    plt.plot(history.history['val_accuracy'], label='Validation')
    plt.title('Training vs Validation Accuracy')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train')
    plt.plot(history.history['val_loss'], label='Validation')
    plt.title('Training vs Validation Loss')
    plt.ylabel('Loss')
    plt.legend()
    
    plot_dir = "plots"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f"{plot_dir}/training_curves_{timestamp}.png", dpi=300)
    plt.close()

plot_training(history)

# Evaluation
y_pred = (model.predict(test_padded) > 0.5).astype(int)
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))