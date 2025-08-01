# buildAndTrainModel.py
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from customImputerLayerDefinition import ImputerLayer

# Create directory for plots
if not os.path.exists('plots'):
    os.makedirs('plots')

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class ModelBuilder:
    def __init__(self):
        self.BATCH_SIZE = 32
        
    def parse_example(self, example_proto):
        """Parse TFRecord example"""
        feature_description = {
            'tickers': tf.io.FixedLenFeature([188], tf.float32),
            'weekday': tf.io.FixedLenFeature([1], tf.int64),
            'hour': tf.io.FixedLenFeature([1], tf.int64),
            'month': tf.io.FixedLenFeature([1], tf.int64),
            'target': tf.io.FixedLenFeature([1], tf.int64)
        }
        
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        
        return ({
            'tickers': parsed_features['tickers'],
            'weekday': tf.math.mod(parsed_features['weekday'], 7),
            'hour': tf.math.mod(parsed_features['hour'], 24),
            'month': tf.math.mod(parsed_features['month'] - 1, 12)
        }, parsed_features['target'])

    def prepare_datasets(self):
        """Prepare training, validation, and test datasets"""
        # Calculate dataset size
        self.dataset_size = sum(1 for _ in tf.data.TFRecordDataset('dataset.tfrecords'))
        print(f"Total dataset size: {self.dataset_size}")

        # Split sizes
        self.train_size = int(0.7 * self.dataset_size)
        self.val_size = int(0.15 * self.dataset_size)
        self.test_size = self.dataset_size - self.train_size - self.val_size

        # Create datasets
        full_dataset = tf.data.TFRecordDataset('dataset.tfrecords')
        full_dataset = full_dataset.map(self.parse_example)
        full_dataset = full_dataset.shuffle(buffer_size=1000)

        # Split datasets
        self.train_dataset = full_dataset.take(self.train_size)
        remaining = full_dataset.skip(self.train_size)
        self.val_dataset = remaining.take(self.val_size)
        self.test_dataset = remaining.skip(self.val_size)

        # Prepare for training
        self.train_dataset = self.train_dataset.batch(self.BATCH_SIZE).cache().repeat()
        self.val_dataset = self.val_dataset.batch(self.BATCH_SIZE).cache().repeat()
        self.test_dataset = self.test_dataset.batch(self.BATCH_SIZE).cache()

        # Calculate steps
        self.steps_per_epoch = self.train_size // self.BATCH_SIZE
        self.validation_steps = self.val_size // self.BATCH_SIZE
        self.test_steps = self.test_size // self.BATCH_SIZE

    def build_model(self):
        """Build and compile the model"""
        # Inputs
        tickers_input = tf.keras.Input(shape=(188,), name='tickers')
        weekday_input = tf.keras.Input(shape=(1,), name='weekday', dtype=tf.int64)
        hour_input = tf.keras.Input(shape=(1,), name='hour', dtype=tf.int64)
        month_input = tf.keras.Input(shape=(1,), name='month', dtype=tf.int64)

        # Preprocessing layers
        imputer = ImputerLayer()
        normalizer = tf.keras.layers.Normalization()

        # Process tickers
        tickers_data = []
        for x, _ in self.train_dataset.take(self.steps_per_epoch):
            tickers_data.append(x['tickers'].numpy())
        tickers_data = np.vstack(tickers_data)

        imputer.adapt(tickers_data)
        normalizer.adapt(imputer(tickers_data))
        processed_tickers = normalizer(imputer(tickers_input))

        # Embeddings
        weekday_embedding = tf.keras.layers.Embedding(7, 3)(weekday_input)
        hour_embedding = tf.keras.layers.Embedding(24, 4)(hour_input)
        month_embedding = tf.keras.layers.Embedding(12, 3)(month_input)

        # Flatten embeddings
        weekday_flat = tf.keras.layers.Flatten()(weekday_embedding)
        hour_flat = tf.keras.layers.Flatten()(hour_embedding)
        month_flat = tf.keras.layers.Flatten()(month_embedding)

        # Combine features
        concat = tf.keras.layers.Concatenate()([processed_tickers, weekday_flat, hour_flat, month_flat])

        # Hidden layers
        x = tf.keras.layers.Dense(128, activation='relu')(concat)
        x = tf.keras.layers.Dropout(0.3)(x)
        x = tf.keras.layers.Dense(64, activation='relu')(x)

        # Output
        outputs = tf.keras.layers.Dense(22, activation='softmax')(x)

        # Create and compile model
        self.model = tf.keras.Model(
            inputs=[tickers_input, weekday_input, hour_input, month_input],
            outputs=outputs
        )

        self.model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def plot_training_history(self, history):
        """Plot and save training metrics"""
        plt.figure(figsize=(12, 4))
        
        # Accuracy plot
        plt.subplot(1, 2, 1)
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        
        # Loss plot
        plt.subplot(1, 2, 2)
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Model Loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Validation'])
        
        plt.tight_layout()
        plt.savefig('plots/training_history.png')
        plt.close()

    def plot_confusion_matrix(self, y_true, y_pred):
        """Plot and save confusion matrix"""
        cm = tf.math.confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(12, 10))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig('plots/confusion_matrix.png')
        plt.close()

    def plot_metrics(self, predictions, true_labels):
        """Plot various metrics"""
        # Prediction distribution
        plt.figure(figsize=(10, 6))
        plt.hist(np.argmax(predictions, axis=1), bins=22, rwidth=0.8)
        plt.title('Prediction Distribution')
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.savefig('plots/prediction_distribution.png')
        plt.close()

        # Class-wise accuracy
        class_correct = np.zeros(22)
        class_total = np.zeros(22)
        pred_classes = np.argmax(predictions, axis=1)

        for true, pred in zip(true_labels, pred_classes):
            class_total[true] += 1
            if true == pred:
                class_correct[true] += 1

        class_accuracy = class_correct / (class_total + 1e-10)

        plt.figure(figsize=(15, 6))
        plt.bar(range(22), class_accuracy)
        plt.title('Class-wise Accuracy')
        plt.xlabel('Class')
        plt.ylabel('Accuracy')
        plt.savefig('plots/class_accuracy.png')
        plt.close()

    def train_and_evaluate(self):
        """Train the model and generate visualizations"""
        try:
            # Prepare data
            self.prepare_datasets()
            
            # Build model
            self.build_model()
            self.model.summary()

            # Callbacks
            callbacks = [
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=5,
                    restore_best_weights=True
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=3
                )
            ]

            # Train model
            print("Starting model training...")
            history = self.model.fit(
                self.train_dataset,
                validation_data=self.val_dataset,
                epochs=40,
                steps_per_epoch=self.steps_per_epoch,
                validation_steps=self.validation_steps,
                callbacks=callbacks,
                verbose=1
            )

            # Save model
            self.model.save('mySavedModel.keras')

            # Generate visualizations
            print("\nGenerating visualizations...")
            self.plot_training_history(history)

            # Get predictions
            test_data = self.test_dataset.take(self.test_steps)
            true_labels = []
            predictions = []

            for x, y in test_data:
                pred = self.model.predict(x, verbose=0)
                predictions.extend(pred)
                true_labels.extend(y.numpy())

            predictions = np.array(predictions)
            true_labels = np.array(true_labels)

            # Generate plots
            self.plot_confusion_matrix(true_labels, np.argmax(predictions, axis=1))
            self.plot_metrics(predictions, true_labels)

            # Evaluate model
            test_results = self.model.evaluate(
                self.test_dataset, 
                steps=self.test_steps, 
                verbose=1
            )

            # Save metrics summary
            with open('plots/metrics_summary.txt', 'w') as f:
                f.write("Model Performance Summary\n")
                f.write("========================\n")
                f.write(f"Test Loss: {test_results[0]:.4f}\n")
                f.write(f"Test Accuracy: {test_results[1]:.4f}\n")

            print("\nTraining and evaluation completed successfully!")
            print("Check the 'plots' directory for visualizations.")

        except Exception as e:
            print(f"An error occurred: {str(e)}")
            import traceback
            print(traceback.format_exc())

if __name__ == "__main__":
    model_builder = ModelBuilder()
    model_builder.train_and_evaluate()