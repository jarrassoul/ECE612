import os
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.layers import (Dense, GlobalAveragePooling2D, Dropout, 
                                   BatchNormalization, RandomRotation, RandomFlip,
                                   RandomZoom, RandomBrightness, RandomContrast,
                                   RandomTranslation)
from tensorflow.keras.metrics import SparseTopKCategoricalAccuracy
from tensorflow.keras.applications import EfficientNetV2B3
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, 
                                      ReduceLROnPlateau, CSVLogger)
from datetime import datetime
import matplotlib.pyplot as plt
import logging
import gc

# Configure environment
os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)

class TrainingDetailsCallback(tf.keras.callbacks.Callback):
    """Custom callback for detailed training progress"""
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        print(f"\nEpoch {epoch + 1} Summary:")
        for metric, value in logs.items():
            print(f"{metric:>12}: {value:.4f}")

class BirdClassifier:
    def __init__(self):
        # Training parameters
        self.batch_size = 16
        self.num_classes = 358
        self.img_size = 300
        self.train_examples_per_class = 20
        self.val_examples_per_class = 10
        
        # Calculate dataset parameters
        self.total_train_examples = self.train_examples_per_class * self.num_classes
        self.total_val_examples = self.val_examples_per_class * self.num_classes
        self.train_steps = int(np.ceil(self.total_train_examples / self.batch_size))
        self.val_steps = int(np.ceil(self.total_val_examples / self.batch_size))
        
        # Setup output directory
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = f'bird_training_{self.timestamp}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Initialize metrics
        self.metrics = [
            tf.keras.metrics.SparseCategoricalAccuracy(name='accuracy'),
            SparseTopKCategoricalAccuracy(k=5, name='top5'),
            SparseTopKCategoricalAccuracy(k=10, name='top10'),
            SparseTopKCategoricalAccuracy(k=20, name='top20')
        ]
        
        # Log configuration
        self._log_configuration()

    def _log_configuration(self):
        """Log training configuration"""
        config_info = {
            "Image size": f"{self.img_size}x{self.img_size}",
            "Batch size": self.batch_size,
            "Number of classes": self.num_classes,
            "Training examples": self.total_train_examples,
            "Validation examples": self.total_val_examples,
            "Training steps": self.train_steps,
            "Validation steps": self.val_steps,
            "Output directory": self.output_dir,
            "Base model": "EfficientNetV2B3",
            "Training strategy": "Two-phase (frozen base + fine-tune)"
        }
        
        print("\nTraining Configuration:")
        print("=" * 50)
        for key, value in config_info.items():
            print(f"{key:>20}: {value}")
            logging.info(f"{key}: {value}")

    def setup_gpu(self):
        """Configure GPU settings"""
        try:
            gpus = tf.config.list_physical_devices('GPU')
            if gpus:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print(f"\nFound {len(gpus)} GPU(s). Configuration completed.")
            else:
                print("\nNo GPU found. Running on CPU")
        except Exception as e:
            print(f"\nGPU setup error: {e}")

    def get_data_augmentation(self):
        """Enhanced data augmentation pipeline"""
        return tf.keras.Sequential([
            RandomFlip("horizontal_and_vertical"),
            RandomRotation(0.25),
            RandomZoom(0.25),
            RandomBrightness(0.25),
            RandomContrast(0.25),
            RandomTranslation(0.15, 0.15),
        ], name='data_augmentation')

    def parse_tfrecord(self, example_proto):
        """Parse TFRecord data with preprocessing"""
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'birdType': tf.io.FixedLenFeature([], tf.int64)
        }
        
        features = tf.io.parse_single_example(example_proto, feature_description)
        
        # Process image
        image = tf.io.decode_jpeg(features['image'], channels=3)
        image = tf.image.resize(image, [self.img_size, self.img_size])
        image = tf.keras.applications.efficientnet_v2.preprocess_input(image)
        
        # Get label
        label = features['birdType']
        
        return image, label

    def prepare_dataset(self, file_path, is_training=True):
        """Optimized dataset pipeline"""
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"TFRecord file not found: {file_path}")
            
            # Create dataset
            dataset = tf.data.TFRecordDataset(file_path, num_parallel_reads=tf.data.AUTOTUNE)
            dataset = dataset.map(self.parse_tfrecord, 
                                 num_parallel_calls=tf.data.AUTOTUNE)
            
            if is_training:
                # Add repeat for training dataset
                dataset = dataset.repeat()
                dataset = dataset.shuffle(2000, reshuffle_each_iteration=True)
                augmentation = self.get_data_augmentation()
                dataset = dataset.map(
                    lambda x, y: (augmentation(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE
                )
            
            dataset = dataset.batch(self.batch_size, drop_remainder=False)
            dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
            
            return dataset
            
        except Exception as e:
            print(f"Dataset preparation error: {e}")
            raise

    def build_model(self):
        """Build optimized model architecture"""
        try:
            # Create base model
            base_model = EfficientNetV2B3(
                include_top=False,
                weights='imagenet',
                input_shape=(self.img_size, self.img_size, 3),
                pooling=None
            )
            base_model.trainable = False
            
            # Create custom head
            inputs = tf.keras.Input(shape=(self.img_size, self.img_size, 3))
            x = base_model(inputs, training=False)
            x = GlobalAveragePooling2D()(x)
            x = Dense(1536, activation='swish')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.5)(x)
            x = Dense(768, activation='swish')(x)
            x = BatchNormalization()(x)
            x = Dropout(0.3)(x)
            outputs = Dense(self.num_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs, outputs)
            
            # Compile model
            model.compile(
                optimizer=Adam(learning_rate=1e-3),
                loss='sparse_categorical_crossentropy',
                metrics=self.metrics
            )
            
            print("\nModel Summary:")
            model.summary()
            
            return model, base_model
            
        except Exception as e:
            print(f"Model building error: {e}")
            return None, None

    def get_callbacks(self):
        """Enhanced training callbacks"""
        return [
            TrainingDetailsCallback(),
            ModelCheckpoint(
                filepath=os.path.join(self.output_dir, 'best_model.keras'),
                monitor='val_top5',
                mode='max',
                save_best_only=True,
                save_weights_only=False,
                verbose=1
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=12,
                restore_best_weights=True,
                verbose=1
            ),
            ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.25,
                patience=6,
                min_lr=1e-7,
                verbose=1
            ),
            CSVLogger(
                os.path.join(self.output_dir, 'training_log.csv')
            )
        ]

    def plot_all_metrics(self, history1, history2=None):
        """Plot all metrics in a single figure with robust history handling"""
        try:
            print("\nCreating consolidated metrics visualization...")
            
            # Create combined history dictionary
            full_history = {}
            
            # Safely combine histories
            for h in [history1, history2]:
                if h is None:
                    continue
                for key, values in h.history.items():
                    if key not in full_history:
                        full_history[key] = []
                    full_history[key].extend(values)
            
            # Create subplots for available metrics
            metrics = ['loss', 'accuracy', 'top5', 'top10', 'top20']
            available_metrics = [m for m in metrics if m in full_history]
            
            if not available_metrics:
                print("No metrics available for plotting")
                return
                
            fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(16, 18))
            axes = axes.flatten()
            
            # Plot each available metric
            for i, metric in enumerate(available_metrics):
                ax = axes[i]
                epochs = range(1, len(full_history[metric]) + 1)
                
                # Plot training metric
                ax.plot(epochs, full_history[metric], 'b-', label=f'Training {metric}')
                
                # Plot validation metric if available
                val_metric = f'val_{metric}'
                if val_metric in full_history:
                    ax.plot(epochs, full_history[val_metric], 'r-', label=f'Validation {metric}')
                
                # Add phase separator if applicable
                if history2:
                    phase_change = len(history1.history[metric])
                    if phase_change < len(epochs):
                        ax.axvline(x=phase_change, color='g', linestyle='--', 
                                  label='Fine-tuning Start')
                
                ax.set_title(f'{metric.upper()} Evolution')
                ax.set_xlabel('Epochs')
                ax.set_ylabel(metric.capitalize())
                ax.legend()
                ax.grid(True)
            
            # Remove unused subplots
            for j in range(len(available_metrics), len(axes)):
                fig.delaxes(axes[j])
            
            # Add overall title
            plt.suptitle('Training Metrics Overview', fontsize=16, y=0.99)
            plt.tight_layout(rect=[0, 0, 1, 0.96])
            
            # Save plot
            plot_path = os.path.join(self.output_dir, 'all_metrics.png')
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved consolidated metrics plot: {plot_path}")
            
        except Exception as e:
            print(f"Error plotting metrics: {e}")
            logging.error(f"Plotting error: {e}")

    def train(self):
        """Robust training pipeline with enhanced error handling"""
        try:
            self.setup_gpu()
            print("\nStarting optimized training pipeline...")

            # Prepare datasets
            train_dataset = self.prepare_dataset('birds-20-eachOf-358.tfrecords', True)
            val_dataset = self.prepare_dataset('birds-10-eachOf-358.tfrecords', False)

            # Build model
            model, base_model = self.build_model()
            if model is None:
                raise Exception("Model building failed")

            # Phase 1: Initial training
            print("\nPHASE 1: Training Classifier Head (Base Frozen)")
            print("=" * 60)
            history1 = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=50,
                steps_per_epoch=self.train_steps,
                validation_steps=self.val_steps,
                callbacks=self.get_callbacks(),
                verbose=1
            )

            # Phase 2: Fine-tuning
            print("\nPHASE 2: Fine-tuning Base Model")
            print("=" * 60)
            
            # Unfreeze and fine-tune top layers
            base_model.trainable = True
            for layer in base_model.layers[:len(base_model.layers)//2]:
                layer.trainable = False
                
            # Recompile with lower learning rate
            model.compile(
                optimizer=Adam(learning_rate=1e-5),
                loss='sparse_categorical_crossentropy',
                metrics=self.metrics
            )

            history2 = model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=40,
                initial_epoch=len(history1.history['loss']),
                steps_per_epoch=self.train_steps,
                validation_steps=self.val_steps,
                callbacks=self.get_callbacks(),
                verbose=1
            )

            # Save final model
            final_model_path = os.path.join(self.output_dir, 'final_model.keras')
            model.save(final_model_path)
            print(f"\nFinal model saved to: {final_model_path}")

            # Plot metrics
            self.plot_all_metrics(history1, history2)

            # Display final metrics with robust history handling
            print("\nFinal Validation Metrics:")
            print("=" * 50)
            
            # Create combined history dictionary
            full_history = {}
            for h in [history1, history2]:
                if h is not None:
                    for key, values in h.history.items():
                        if key not in full_history:
                            full_history[key] = []
                        full_history[key].extend(values)
            
            # Safely extract final validation metrics
            for metric_name in ['accuracy', 'top5', 'top10', 'top20']:
                val_key = f'val_{metric_name}'
                if val_key in full_history and full_history[val_key]:
                    final_value = full_history[val_key][-1]
                    print(f"{metric_name:>12}: {final_value:.4f}")
                else:
                    print(f"{metric_name:>12}: Metric not available")

        except Exception as e:
            print(f"\nTraining error: {e}")
            logging.error(f"Training failed: {e}", exc_info=True)
        finally:
            gc.collect()
            tf.keras.backend.clear_session()

def main():
    """Main execution"""
    try:
        print("\nStarting Bird Classification Training")
        print("=" * 50)
        classifier = BirdClassifier()
        classifier.train()
    except Exception as e:
        print(f"\nMain execution error: {e}")
        logging.error(f"Main execution failed: {e}", exc_info=True)
    finally:
        gc.collect()
        tf.keras.backend.clear_session()

if __name__ == "__main__":
    main()