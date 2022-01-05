import tensorflow as tf
import numpy as np
import pandas as pd
import joblib
from google.cloud import storage

import os


BUCKET_NAME = "wagon-data-779-createur_de_recette"
BUCKET_DATA_PATH="data/train_data.csv"
MODEL_VERSION = 'v1'
STORAGE_LOCATION = 'models/'

N_ROWS = 100             # Number of rows. None for the full dataset
STOP_SIGN = '␣'          # Used for padding
MAX_RECIPE_LENGTH = 1000 # For padding
EMBEDDING_DIM = 256
RNN_UNITS = 1024
BATCH_SIZE = 64
SHUFFLE_BUFFER_SIZE = 1000
EPOCHS = 500
INITIAL_EPOCH = 1
STEPS_PER_EPOCH = 1500

class Trainer:
    def __init__(self, on_gcp=False):
        self.tokenizer = tf.keras.preprocessing.text.Tokenizer(
            char_level=True,
            filters='',
            lower=False,
            split=''
        )
        self.vocabulary_size = 0
        self.on_gcp = on_gcp
        self.dataset_train = None
        self.dataset_filtered = None
        
    def load_data(self, nrows=None):
        prefix = f"gs://{BUCKET_NAME}/" if self.on_gcp else ""
        self.dataset_filtered = pd.read_csv(prefix + "data/train_data.csv", header=None, nrows=nrows)[0]
        

    def recipe_sequence_to_string(self, recipe_sequence):
        recipe_stringified = self.tokenizer.sequences_to_texts([recipe_sequence])[0]
        print(recipe_stringified)

    def split_input_target(self, recipe):
        input_text = recipe[:-1]
        target_text = recipe[1:]
        return input_text, target_text

    def build_model(self, batch_size=BATCH_SIZE):
        if self.dataset_train is None:
            print("Data not tokenized. Call tokenize() first")
        model = tf.keras.models.Sequential()

        model.add(tf.keras.layers.Embedding(
            input_dim=self.vocabulary_size,
            output_dim=EMBEDDING_DIM,
            batch_input_shape=[batch_size, None]
        ))

        model.add(tf.keras.layers.LSTM(
            units=RNN_UNITS,
            return_sequences=True,
            stateful=True,
            recurrent_initializer=tf.keras.initializers.GlorotNormal()
        ))

        model.add(tf.keras.layers.Dense(self.vocabulary_size))
        
        return model

    def loss(self, labels, logits):
        entropy = tf.keras.losses.sparse_categorical_crossentropy(
            y_true=labels,
            y_pred=logits,
            from_logits=True
        )
        
        return entropy
    
    def tokenize(self):
        if self.dataset_filtered is None:
            print("Data not loaded. Call load_data() first")
        self.tokenizer.fit_on_texts([STOP_SIGN])
        self.tokenizer.fit_on_texts(self.dataset_filtered)
        self.vocabulary_size = len(self.tokenizer.word_counts) + 1
        dataset_vectorized = self.tokenizer.texts_to_sequences(self.dataset_filtered)
        
        dataset_vectorized_padded_without_stops = tf.keras.preprocessing.sequence.pad_sequences(
            dataset_vectorized,
            padding='post',
            truncating='post',
            # We use -1 here and +1 in the next step to make sure
            # that all recipes will have at least 1 stops sign at the end,
            # since each sequence will be shifted and truncated afterwards
            # (to generate X and Y sequences).
            maxlen=MAX_RECIPE_LENGTH-1,
            value=self.tokenizer.texts_to_sequences([STOP_SIGN])[0]
        )

        dataset_vectorized_padded = tf.keras.preprocessing.sequence.pad_sequences(
            dataset_vectorized_padded_without_stops,
            padding='post',
            truncating='post',
            maxlen=MAX_RECIPE_LENGTH+1,
            value=self.tokenizer.texts_to_sequences([STOP_SIGN])[0]
        )
        
        dataset = tf.data.Dataset.from_tensor_slices(dataset_vectorized_padded)
        dataset_targeted = dataset.map(self.split_input_target)
        self.dataset_train = dataset_targeted.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True).repeat()
        
    

    def get_model(self):
        if self.dataset_train is None:
            print("Data not tokenized. Call tokenize() first")
        model = self.build_model(BATCH_SIZE)
        
        for input_example_batch, target_example_batch in self.dataset_train.take(1):
            example_batch_predictions = model(input_example_batch)
            
        sampled_indices = tf.random.categorical(
            logits=example_batch_predictions[0],
            num_samples=1
        )

        sampled_indices = tf.squeeze(
            input=sampled_indices,
            axis=-1
        ).numpy()
        
        adam_optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
        
        model.compile(
            optimizer=adam_optimizer,
            loss=self.loss
        )
        
        early_stopping_callback = tf.keras.callbacks.EarlyStopping(
            patience=5,
            monitor='loss',
            restore_best_weights=True,
            verbose=1
        )
        
        checkpoint_dir = 'tmp/checkpoints'
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt_{epoch}')
        checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_prefix,
            save_weights_only=True
        )
        
        model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        
        history = model.fit(
            x=self.dataset_train,
            epochs=EPOCHS,
            steps_per_epoch=STEPS_PER_EPOCH,
            initial_epoch=INITIAL_EPOCH,
            callbacks=[
                checkpoint_callback,
                early_stopping_callback
            ]
        )
        
        # Saving the trained model to file (to be able to re-use it later).
        model_name = 'recipe_generation_rnn_raw.h5'
        model.save(model_name, save_format='h5')
        
        simplified_batch_size = 1

        model_simplified = self.build_model(simplified_batch_size)
        model_simplified.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
        model_simplified.build(tf.TensorShape([simplified_batch_size, None]))

        return model_simplified

    def generate_text(self, model, start_string, num_generate = 1000, temperature=1.0):
        # Evaluation step (generating text using the learned model)
        padded_start_string = start_string

        # Converting our start string to numbers (vectorizing).
        input_indices = np.array(self.tokenizer.texts_to_sequences([padded_start_string]))

        # Empty string to store our results.
        text_generated = []

        # Here batch size == 1.
        model.reset_states()
        for char_index in range(num_generate):
            predictions = model(input_indices)
            # remove the batch dimension
            predictions = tf.squeeze(predictions, 0)

            # Using a categorical distribution to predict the character returned by the model.
            predictions = predictions / temperature
            predicted_id = tf.random.categorical(
                predictions,
                num_samples=1
            )[-1, 0].numpy()

            # We pass the predicted character as the next input to the model
            # along with the previous hidden state.
            input_indices = tf.expand_dims([predicted_id], 0)
            
            next_character = self.tokenizer.sequences_to_texts(input_indices.numpy())[0]

            text_generated.append(next_character)

        return (padded_start_string + ''.join(text_generated))

    def generate_combinations(self, model, try_letters, try_temperature):
        recipe_length = 1000

        for letter in try_letters:
            for temperature in try_temperature:
                generated_text = self.generate_text(
                    model,
                    start_string=letter,
                    num_generate = recipe_length,
                    temperature=temperature
                )
                print(f'Attempt: "{letter}" + {temperature}')
                print('-----------------------------------')
                print(generated_text)
                print('\n\n')
                
    def save_model(self, reg):
        """method that saves the model into a .joblib file and uploads it on Google Storage /models folder
        HINTS : use joblib library and google-cloud-storage"""
        file_name = f"model.jolib"
        joblib.dump(reg, file_name)
        print("saved model.joblib locally")
        upload_model_to_gcp(file_name)
        print(f"uploaded model.joblib to gcp cloud storage under \n => {STORAGE_LOCATION}")


    def upload_model_to_gcp(self, file_name):
        client = storage.Client()
        bucket = client.bucket(BUCKET_NAME)
        blob = bucket.blob(STORAGE_LOCATION + file_name)
        blob.upload_from_filename(file_name)


if __name__ == "__main__":
    trainer = Trainer(on_gcp=False)
    trainer.load_data()
    trainer.tokenize()
    trainer.get_model()
    #save_model(model)
    trainer.generate_combinations(model, ['🥕', 'Pomme', 'Manger', 'Plat', 'Le', 'Banane'], [1.0, 0.8, 0.4, 0.2])
