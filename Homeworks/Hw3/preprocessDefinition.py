
# import tensorflow as tf
# import numpy as np
# from datetime import datetime
# import logging
# import matplotlib.pyplot as plt
# import seaborn as sns
# import os

# class BirdClassifier:
#     def __init__(self):
#         # Create timestamp for this training run
#         self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        
#         # Create output directory
#         self.output_dir = f'Bird_training_output_{self.timestamp}'
#         os.makedirs(self.output_dir, exist_ok=True)
        
#         # Setup logging first
#         self.setup_logging()
        
#         # Load bird names
#         with open('birdNames.txt', 'r') as f:
#             self.bird_names = f.read().splitlines()
        
#         # Training parameters
#         self.batch_size = 32  # Increased batch size
#         self.initial_learning_rate = 1e-4
#         self.num_classes = len(self.bird_names)
        
#         # Dataset parameters (will be set in create_datasets)
#         self.train_size = 0
#         self.val_size = 0
#         self.steps_per_epoch = 0
#         self.validation_steps = 0
        
#         # Configure GPU
#         self.setup_gpu()

#     def setup_logging(self):
#         """Configure logging settings"""
#         logging.basicConfig(
#             filename=os.path.join(self.output_dir, 'training.log'),
#             level=logging.INFO,
#             format='%(asctime)s - %(levelname)s - %(message)s'
#         )
#         # Also print to console
#         console_handler = logging.StreamHandler()
#         console_handler.setLevel(logging.INFO)
#         logging.getLogger().addHandler(console_handler)

#     def setup_gpu(self):
#         """Configure GPU settings"""
#         gpus = tf.config.experimental.list_physical_devices('GPU')
#         if gpus:
#             try:
#                 for gpu in gpus:
#                     tf.config.experimental.set_memory_growth(gpu, True)
#                 logging.info(f"GPU setup completed. Found {len(gpus)} GPU(s)")
#             except RuntimeError as e:
#                 logging.error(f"GPU setup failed: {e}")
#         else:
#             logging.warning("No GPUs found. Running on CPU.")

#     def parse_and_preprocess(self, serialized_example, is_training=True):
#         """Parse and preprocess a single example"""
#         feature_description = {
#             'image': tf.io.FixedLenFeature([], tf.string),
#             'birdType': tf.io.FixedLenFeature([], tf.int64)
#         }
        
#         example = tf.io.parse_single_example(serialized_example, feature_description)
        
#         # Decode and preprocess image
#         image = tf.io.decode_jpeg(example['image'], channels=3)
#         image = tf.cast(image, tf.float32)
        
#         # Normalize the image to [0,1] range
#         image = image / 255.0
        
#         # Resize image
#         image = tf.image.resize(image, [224, 224])
        
#         if is_training:
#             # Data augmentation for training
#             image = tf.image.random_flip_left_right(image)
#             image = tf.image.random_brightness(image, 0.2)
#             image = tf.image.random_contrast(image, 0.8, 1.2)
#             image = tf.image.random_saturation(image, 0.8, 1.2)
#             image = tf.clip_by_value(image, 0.0, 1.0)
        
#         # Ensure the label is properly formatted
#         label = tf.cast(example['birdType'], tf.int32)
        
#         return image, label

#     def create_datasets(self):
#         """Create and prepare training and validation datasets"""
#         try:
#             # Load datasets and count samples
#             train_ds = tf.data.TFRecordDataset(['birds-20-eachOf-358.tfrecords'])
#             val_ds = tf.data.TFRecordDataset(['birds-10-eachOf-358.tfrecords'])

#             # Calculate dataset sizes
#             self.train_size = sum(1 for _ in train_ds)
#             self.val_size = sum(1 for _ in val_ds)
            
#             logging.info(f"Training dataset size: {self.train_size}")
#             logging.info(f"Validation dataset size: {self.val_size}")

#             # Reset datasets after counting
#             train_ds = tf.data.TFRecordDataset(['birds-20-eachOf-358.tfrecords'])
#             val_ds = tf.data.TFRecordDataset(['birds-10-eachOf-358.tfrecords'])

#             # Create training dataset
#             train_ds = train_ds.map(
#                 lambda x: self.parse_and_preprocess(x, True),
#                 num_parallel_calls=tf.data.AUTOTUNE
#             )
#             train_ds = train_ds.shuffle(buffer_size=self.train_size)
#             train_ds = train_ds.batch(self.batch_size)
#             train_ds = train_ds.repeat()  # Add repeat for infinite iterations
#             train_ds = train_ds.prefetch(tf.data.AUTOTUNE)

#             # Create validation dataset
#             val_ds = val_ds.map(
#                 lambda x: self.parse_and_preprocess(x, False),
#                 num_parallel_calls=tf.data.AUTOTUNE
#             )
#             val_ds = val_ds.batch(self.batch_size)
#             val_ds = val_ds.repeat()  # Add repeat for infinite iterations
#             val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

#             # Calculate steps per epoch
#             self.steps_per_epoch = self.train_size // self.batch_size
#             self.validation_steps = self.val_size // self.batch_size

#             # Ensure at least one step
#             self.steps_per_epoch = max(1, self.steps_per_epoch)
#             self.validation_steps = max(1, self.validation_steps)

#             logging.info(f"Steps per epoch: {self.steps_per_epoch}")
#             logging.info(f"Validation steps: {self.validation_steps}")

#             # Validate the first batch
#             for images, labels in train_ds.take(1):
#                 logging.info(f"Batch shape: {images.shape}")
#                 logging.info(f"Labels shape: {labels.shape}")
#                 logging.info(f"Image value range: [{tf.reduce_min(images)}, {tf.reduce_max(images)}]")
#                 logging.info(f"Sample labels: {labels[:5]}")

#             return train_ds, val_ds

#         except Exception as e:
#             logging.error(f"Error creating datasets: {str(e)}")
#             raise

#     def create_model(self):
#         """Create the model architecture"""
#         try:
#             base_model = tf.keras.applications.ResNet50(
#                 include_top=False,
#                 weights='imagenet',
#                 input_shape=(224, 224, 3)
#             )
#             base_model.trainable = False

#             inputs = tf.keras.Input(shape=(224, 224, 3))
#             x = base_model(inputs)
#             x = tf.keras.layers.GlobalAveragePooling2D()(x)
#             x = tf.keras.layers.BatchNormalization()(x)
#             x = tf.keras.layers.Dense(512, activation='relu')(x)
#             x = tf.keras.layers.Dropout(0.5)(x)
#             x = tf.keras.layers.Dense(256, activation='relu')(x)
#             x = tf.keras.layers.Dropout(0.3)(x)
#             outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)

#             model = tf.keras.Model(inputs, outputs)

#             logging.info("Model created successfully")
#             model.summary(print_fn=logging.info)
#             return model, base_model

#         except Exception as e:
#             logging.error(f"Error creating model: {str(e)}")
#             raise
    
#     def create_training_plots(self, history1, history2):
#         """Create and save training visualization plots"""
#         try:
#             # Combine histories
#             combined_history = {
#                 'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
#                 'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
#                 'loss': history1.history['loss'] + history2.history['loss'],
#                 'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
#                 'top5': history1.history['top5'] + history2.history['top5'],
#                 'val_top5': history1.history['val_top5'] + history2.history['val_top5']
#             }
            
#             # Set the style using seaborn
#             sns.set_style("darkgrid")
#             sns.set_context("notebook", font_scale=1.2)
            
#             fig = plt.figure(figsize=(15, 10))
            
#             # Create plots
#             self._create_accuracy_plot(combined_history, history1)
#             self._create_loss_plot(combined_history, history1)
#             self._create_top5_plot(combined_history, history1)
#             self._create_summary_plot(history1, history2)
            
#             # Save plot
#             plot_path = os.path.join(self.output_dir, 'training_plots.png')
#             plt.tight_layout()
#             plt.savefig(plot_path, dpi=300, bbox_inches='tight')
#             plt.close()
            
#             return plot_path

#         except Exception as e:
#             logging.error(f"Error creating plots: {str(e)}")
#             raise

#     def _create_accuracy_plot(self, combined_history, history1):
#         plt.subplot(2, 2, 1)
#         plt.plot(combined_history['accuracy'], label='Training', color='blue', linewidth=2)
#         plt.plot(combined_history['val_accuracy'], label='Validation', color='orange', linewidth=2)
#         plt.axvline(x=len(history1.history['accuracy'])-1, color='red', linestyle='--', 
#                    label='Fine-tuning Start')
#         plt.title('Model Accuracy Over Time')
#         plt.xlabel('Epoch')
#         plt.ylabel('Accuracy')
#         plt.legend()
#         plt.grid(True, alpha=0.3)

#     def _create_loss_plot(self, combined_history, history1):
#         plt.subplot(2, 2, 2)
#         plt.plot(combined_history['loss'], label='Training', color='blue', linewidth=2)
#         plt.plot(combined_history['val_loss'], label='Validation', color='orange', linewidth=2)
#         plt.axvline(x=len(history1.history['loss'])-1, color='red', linestyle='--', 
#                    label='Fine-tuning Start')
#         plt.title('Model Loss Over Time')
#         plt.xlabel('Epoch')
#         plt.ylabel('Loss')
#         plt.legend()
#         plt.grid(True, alpha=0.3)

#     def _create_top5_plot(self, combined_history, history1):
#         plt.subplot(2, 2, 3)
#         plt.plot(combined_history['top5'], label='Training', color='blue', linewidth=2)
#         plt.plot(combined_history['val_top5'], label='Validation', color='orange', linewidth=2)
#         plt.axvline(x=len(history1.history['top5'])-1, color='red', linestyle='--', 
#                    label='Fine-tuning Start')
#         plt.title('Top-5 Accuracy Over Time')
#         plt.xlabel('Epoch')
#         plt.ylabel('Top-5 Accuracy')
#         plt.legend()
#         plt.grid(True, alpha=0.3)

#     def _create_summary_plot(self, history1, history2):
#         plt.subplot(2, 2, 4)
#         summary_text = (
#             f"Training Summary\n\n"
#             f"Initial Training:\n"
#             f"Best Accuracy: {max(history1.history['val_accuracy']):.4f}\n"
#             f"Best Top-5: {max(history1.history['val_top5']):.4f}\n"
#             f"Final Loss: {history1.history['val_loss'][-1]:.4f}\n\n"
#             f"After Fine-tuning:\n"
#             f"Best Accuracy: {max(history2.history['val_accuracy']):.4f}\n"
#             f"Best Top-5: {max(history2.history['val_top5']):.4f}\n"
#             f"Final Loss: {history2.history['val_loss'][-1]:.4f}"
#         )
#         plt.text(0.5, 0.5, summary_text, ha='center', va='center', 
#                 transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.8))
#         plt.axis('off')

#     def train(self):
#         """Train the model with initial training and fine-tuning phases"""
#         try:
#             train_ds, val_ds = self.create_datasets()
#             model, base_model = self.create_model()
            
#             model_path = os.path.join(self.output_dir, 'best_model.keras')

#             # Learning rate schedule with warmup
#             def lr_schedule(epoch):
#                 if epoch < 3:  # Warmup period
#                     return self.initial_learning_rate * (epoch + 1) / 3
#                 return self.initial_learning_rate * (0.1 ** (epoch // 5))

#             callbacks = [
#                 tf.keras.callbacks.EarlyStopping(
#                     monitor='val_accuracy',
#                     patience=15,
#                     restore_best_weights=True,
#                     verbose=1
#                 ),
#                 tf.keras.callbacks.ModelCheckpoint(
#                     model_path,
#                     monitor='val_accuracy',
#                     save_best_only=True,
#                     verbose=1
#                 ),
#                 tf.keras.callbacks.ReduceLROnPlateau(
#                     monitor='val_loss',
#                     factor=0.5,
#                     patience=7,
#                     min_lr=1e-6,
#                     verbose=1
#                 ),
#                 tf.keras.callbacks.LearningRateScheduler(lr_schedule, verbose=1)
#             ]

#             # Initial training phase
#             initial_epochs = 15  # Reduced number of epochs
#             logging.info(f"Starting initial training phase for {initial_epochs} epochs...")
#             logging.info(f"Training with {self.steps_per_epoch} steps per epoch")
#             logging.info(f"Validating with {self.validation_steps} steps per epoch")

#             model.compile(
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate),
#                 loss='sparse_categorical_crossentropy',
#                 metrics=[
#                     'accuracy',
#                     tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')
#                 ]
#             )

#             history1 = model.fit(
#                 train_ds,
#                 validation_data=val_ds,
#                 epochs=initial_epochs,
#                 steps_per_epoch=self.steps_per_epoch,
#                 validation_steps=self.validation_steps,
#                 callbacks=callbacks,
#                 verbose=1
#             )

#             # Fine-tuning phase
#             fine_tune_epochs = 10  # Reduced number of epochs
#             logging.info(f"Starting fine-tuning phase for {fine_tune_epochs} epochs...")
            
#             base_model.trainable = True
#             for layer in base_model.layers[:-30]:
#                 layer.trainable = False

#             model.compile(
#                 optimizer=tf.keras.optimizers.Adam(learning_rate=self.initial_learning_rate/10),
#                 loss='sparse_categorical_crossentropy',
#                 metrics=[
#                     'accuracy',
#                     tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5')
#                 ]
#             )

#             history2 = model.fit(
#                 train_ds,
#                 validation_data=val_ds,
#                 epochs=fine_tune_epochs,
#                 steps_per_epoch=self.steps_per_epoch,
#                 validation_steps=self.validation_steps,
#                 callbacks=callbacks,
#                 verbose=1
#             )

#             plot_path = self.create_training_plots(history1, history2)
            
#             return model, history1, history2, model_path, plot_path

#         except Exception as e:
#             logging.error(f"Error during training: {str(e)}")
#             raise

# def main():
#     try:
#         # Set random seeds for reproducibility
#         tf.random.set_seed(42)
#         np.random.seed(42)
        
#         classifier = BirdClassifier()
#         model, history1, history2, model_path, plot_path = classifier.train()
        
#         final_val_acc = max(history2.history['val_accuracy'])
#         final_val_top5 = max(history2.history['val_top5'])
        
#         logging.info("\nTraining completed successfully!")
#         logging.info(f"Final validation accuracy: {final_val_acc:.4f}")
#         logging.info(f"Final validation top-5 accuracy: {final_val_top5:.4f}")
#         logging.info(f"Best model saved at: {model_path}")
#         logging.info(f"Training plots saved at: {plot_path}")

#     except Exception as e:
#         logging.error(f"Training failed: {str(e)}")
#         raise

# if __name__ == "__main__":
#     main()









# Part 1: Imports and Basic Setup
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
import os
from sklearn.utils.class_weight import compute_class_weight

# Set random seeds for reproducibility
tf.random.set_seed(42)
np.random.seed(42)

class BirdClassifier:
    def __init__(self):
        # Basic setup
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        self.output_dir = f'bird_training_{self.timestamp}'
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Configure logging
        self.setup_logging()
        
        # Load bird names
        with open('birdNames.txt', 'r') as f:
            self.bird_names = f.read().splitlines()
        
        # Model parameters
        self.image_size = (384, 384)
        self.batch_size = 16
        self.num_classes = len(self.bird_names)
        self.initial_learning_rate = 1e-3
        
        # Configure GPU
        self.setup_gpu()
        
    def setup_logging(self):
        """Configure logging settings"""
        logging.basicConfig(
            filename=os.path.join(self.output_dir, 'training.log'),
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        logging.getLogger().addHandler(console_handler)
        
    def setup_gpu(self):
        """Configure GPU settings"""
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                logging.info(f"GPU setup completed. Found {len(gpus)} GPU(s)")
            except RuntimeError as e:
                logging.error(f"GPU setup failed: {e}")
        else:
            logging.warning("No GPUs found. Running on CPU.")

    def mixup_augmentation(self, images, labels, alpha=0.2):
        """Apply mixup augmentation"""
        batch_size = tf.shape(images)[0]
        indices = tf.random.shuffle(tf.range(batch_size))
        
        shuffled_images = tf.gather(images, indices)
        shuffled_labels = tf.gather(labels, indices)
        
        lambda_ = tf.random.uniform([], 0, alpha)
        mixed_images = lambda_ * images + (1 - lambda_) * shuffled_images
        mixed_labels = lambda_ * tf.one_hot(labels, self.num_classes) + \
                      (1 - lambda_) * tf.one_hot(shuffled_labels, self.num_classes)
        
        return mixed_images, mixed_labels

    def augment_image(self, image):
        """Apply various augmentations to the image"""
        # Random flip
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
        
        # Color augmentation
        image = tf.image.random_brightness(image, 0.2)
        image = tf.image.random_contrast(image, 0.8, 1.2)
        image = tf.image.random_saturation(image, 0.8, 1.2)
        image = tf.image.random_hue(image, 0.1)
        
        # Random rotation
        image = tfa.image.rotate(image, tf.random.uniform([], -0.2, 0.2))
        
        # Random zoom
        image = tf.image.random_crop(
            tf.image.resize_with_crop_or_pad(
                image, 
                int(self.image_size[0]*1.2), 
                int(self.image_size[1]*1.2)
            ),
            self.image_size + (3,)
        )
        
        # Ensure values are in valid range
        image = tf.clip_by_value(image, 0, 1)
        
        return image

    def parse_and_preprocess(self, serialized_example, is_training=True):
        """Parse and preprocess a single example"""
        feature_description = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'birdType': tf.io.FixedLenFeature([], tf.int64)
        }
        
        # Parse example
        example = tf.io.parse_single_example(serialized_example, feature_description)
        
        # Decode and preprocess image
        image = tf.io.decode_jpeg(example['image'], channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, self.image_size)
        image = image / 255.0  # Normalize to [0,1]
        
        # Apply augmentation during training
        if is_training:
            image = self.augment_image(image)
        
        # Get label
        label = tf.cast(example['birdType'], tf.int32)
        
        return image, label

    def create_datasets(self):
        """Create and prepare training and validation datasets"""
        try:
            # Load datasets
            train_ds = tf.data.TFRecordDataset(['birds-20-eachOf-358.tfrecords'])
            val_ds = tf.data.TFRecordDataset(['birds-10-eachOf-358.tfrecords'])
            
            # Calculate dataset sizes
            train_size = sum(1 for _ in train_ds)
            val_size = sum(1 for _ in val_ds)
            
            logging.info(f"Training dataset size: {train_size}")
            logging.info(f"Validation dataset size: {val_size}")
            
            # Reset datasets
            train_ds = tf.data.TFRecordDataset(['birds-20-eachOf-358.tfrecords'])
            val_ds = tf.data.TFRecordDataset(['birds-10-eachOf-358.tfrecords'])
            
            # Prepare training dataset
            train_ds = train_ds.map(
                lambda x: self.parse_and_preprocess(x, True),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            train_ds = train_ds.shuffle(5000)
            train_ds = train_ds.batch(self.batch_size)
            train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
            
            # Prepare validation dataset
            val_ds = val_ds.map(
                lambda x: self.parse_and_preprocess(x, False),
                num_parallel_calls=tf.data.AUTOTUNE
            )
            val_ds = val_ds.batch(self.batch_size)
            val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
            
            return train_ds, val_ds, train_size, val_size
            
        except Exception as e:
            logging.error(f"Error creating datasets: {str(e)}")
            raise

    def create_model(self):
        """Create the model architecture"""
        try:
            # Create base model
            base_model = tf.keras.applications.EfficientNetV2B0(
                include_top=False,
                weights='imagenet',
                input_shape=(*self.image_size, 3)
            )
            base_model.trainable = False
            
            # Create model
            inputs = tf.keras.Input(shape=(*self.image_size, 3))
            x = base_model(inputs)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            x = tf.keras.layers.BatchNormalization()(x)
            
            # Add dense layers with dropout
            x = tf.keras.layers.Dense(1024, activation='selu')(x)
            x = tf.keras.layers.Dropout(0.5)(x)
            x = tf.keras.layers.Dense(512, activation='selu')(x)
            x = tf.keras.layers.Dropout(0.3)(x)
            
            # Output layer
            outputs = tf.keras.layers.Dense(self.num_classes, activation='softmax')(x)
            
            model = tf.keras.Model(inputs, outputs)
            
            return model, base_model
            
        except Exception as e:
            logging.error(f"Error creating model: {str(e)}")
            raise

    def train(self):
        """Train the model"""
        try:
            # Create datasets and model
            train_ds, val_ds, train_size, val_size = self.create_datasets()
            model, base_model = self.create_model()
            
            # Calculate steps per epoch
            steps_per_epoch = train_size // self.batch_size
            validation_steps = val_size // self.batch_size
            
            # Create callbacks
            callbacks = [
                tf.keras.callbacks.ModelCheckpoint(
                    os.path.join(self.output_dir, 'best_model.h5'),
                    monitor='val_accuracy',
                    save_best_only=True,
                    verbose=1
                ),
                tf.keras.callbacks.EarlyStopping(
                    monitor='val_accuracy',
                    patience=10,
                    restore_best_weights=True,
                    verbose=1
                ),
                tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.2,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                ),
                tf.keras.callbacks.TensorBoard(
                    log_dir=os.path.join(self.output_dir, 'logs'),
                    histogram_freq=1
                )
            ]
            
            # Initial training phase
            logging.info("Starting initial training phase...")
            model.compile(
                optimizer=tfa.optimizers.AdamW(
                    learning_rate=self.initial_learning_rate,
                    weight_decay=0.0001
                ),
                loss='sparse_categorical_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5'),
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top10')
                ]
            )
            
            history1 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=30,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps
            )
            
            # Fine-tuning phase
            logging.info("Starting fine-tuning phase...")
            base_model.trainable = True
            for layer in base_model.layers[:-50]:
                layer.trainable = False
                
            model.compile(
                optimizer=tfa.optimizers.AdamW(
                    learning_rate=self.initial_learning_rate/10,
                    weight_decay=0.0001
                ),
                loss='sparse_categorical_crossentropy',
                metrics=[
                    'accuracy',
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5, name='top5'),
                    tf.keras.metrics.SparseTopKCategoricalAccuracy(k=10, name='top10')
                ]
            )
            
            history2 = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=20,
                callbacks=callbacks,
                steps_per_epoch=steps_per_epoch,
                validation_steps=validation_steps
            )
            
            # Save final model
            model.save(os.path.join(self.output_dir, 'final_model'))
            
            # Create and save training plots
            self.create_training_plots(history1, history2)
            
            return model, history1, history2
            
        except Exception as e:
            logging.error(f"Error during training: {str(e)}")
            raise

    def create_training_plots(self, history1, history2):
        """Create and save training visualization plots"""
        try:
            # Combine histories
            combined_history = {
                'accuracy': history1.history['accuracy'] + history2.history['accuracy'],
                'val_accuracy': history1.history['val_accuracy'] + history2.history['val_accuracy'],
                'loss': history1.history['loss'] + history2.history['loss'],
                'val_loss': history1.history['val_loss'] + history2.history['val_loss'],
                'top5': history1.history['top5'] + history2.history['top5'],
                'val_top5': history1.history['val_top5'] + history2.history['val_top5']
            }
            
            # Create plots
            plt.figure(figsize=(15, 10))
            
            # Accuracy plot
            plt.subplot(2, 2, 1)
            plt.plot(combined_history['accuracy'], label='Training')
            plt.plot(combined_history['val_accuracy'], label='Validation')
            plt.title('Model Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Accuracy')
            plt.legend()
            
            # Loss plot
            plt.subplot(2, 2, 2)
            plt.plot(combined_history['loss'], label='Training')
            plt.plot(combined_history['val_loss'], label='Validation')
            plt.title('Model Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            
            # Top-5 accuracy plot
            plt.subplot(2, 2, 3)
            plt.plot(combined_history['top5'], label='Training')
            plt.plot(combined_history['val_top5'], label='Validation')
            plt.title('Top-5 Accuracy')
            plt.xlabel('Epoch')
            plt.ylabel('Top-5 Accuracy')
            plt.legend()
            
            # Save plot
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'training_history.png'))
            plt.close()
            
        except Exception as e:
            logging.error(f"Error creating plots: {str(e)}")
            raise

def main():
    try:
        # Create and train classifier
        classifier = BirdClassifier()
        model, history1, history2 = classifier.train()
        
        # Print final results
        logging.info("\nTraining completed successfully!")
        logging.info(f"Final validation accuracy: {max(history2.history['val_accuracy']):.4f}")
        logging.info(f"Final validation top-5 accuracy: {max(history2.history['val_top5']):.4f}")
        
    except Exception as e:
        logging.error(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()