# # buildAndTrainBirdsVsSquirrels.py
# import tensorflow as tf
# import matplotlib.pyplot as plt
# import json
# import os
# import numpy as np
# from sklearn.metrics import classification_report, confusion_matrix

# class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
#     def __init__(self, initial_lr, warmup_epochs, total_epochs):
#         super(WarmUpCosineDecayScheduler, self).__init__()
#         self.initial_lr = initial_lr
#         self.warmup_epochs = warmup_epochs
#         self.total_epochs = total_epochs
        
#     def on_epoch_begin(self, epoch, logs=None):
#         if epoch < self.warmup_epochs:
#             lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
#         else:
#             progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
#             lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
#         self.model.optimizer.learning_rate.assign(lr)

# def create_dataset(file_path, batch_size=32, is_training=True):
#     feature_description = {
#         'image': tf.io.FixedLenFeature([], tf.string),
#         'label': tf.io.FixedLenFeature([], tf.int64)
#     }

#     def _parse_function(example_proto):
#         parsed_features = tf.io.parse_single_example(example_proto, feature_description)
#         image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
#         image = tf.cast(image, tf.float32)
#         image = tf.image.resize_with_pad(image, 224, 224)
#         image = image / 255.0
#         label = parsed_features['label']
#         return image, label

#     def augment(image, label):
#         if is_training:
#             image = tf.image.random_flip_left_right(image)
#             image = tf.image.random_flip_up_down(image)
#             image = tf.image.random_brightness(image, 0.3)
#             image = tf.image.random_contrast(image, 0.7, 1.3)
#             image = tf.image.random_saturation(image, 0.7, 1.3)
#             image = tf.image.random_hue(image, 0.2)
#             image = tf.image.resize(image, [240, 240])
#             image = tf.image.random_crop(image, [224, 224, 3])
#             noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
#             image = image + noise
#             image = tf.clip_by_value(image, 0.0, 1.0)
#         return image, label

#     try:
#         dataset = tf.data.TFRecordDataset(file_path)
#         count = sum(1 for _ in dataset)
#         print(f"Found {count} examples in {file_path}")

#         dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
#         dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

#         if is_training:
#             dataset = dataset.shuffle(1000)
#             dataset = dataset.repeat()

#         dataset = dataset.batch(batch_size)
#         dataset = dataset.prefetch(tf.data.AUTOTUNE)

#         return dataset, count

#     except Exception as e:
#         print(f"Error creating dataset from {file_path}: {e}")
#         raise

# def create_model(num_classes):
#     base_model = tf.keras.applications.MobileNetV2(
#         input_shape=(224, 224, 3),
#         include_top=False,
#         weights='imagenet'
#     )
#     base_model.trainable = False

#     model = tf.keras.Sequential([
#         tf.keras.layers.Resizing(224, 224),
#         base_model,
#         tf.keras.layers.GlobalAveragePooling2D(),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Dropout(0.5),
#         tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
#         tf.keras.layers.BatchNormalization(),
#         tf.keras.layers.Activation('relu'),
#         tf.keras.layers.Dropout(0.4),
#         tf.keras.layers.Dense(num_classes, activation='softmax')
#     ])
    
#     return model

# def plot_metrics(history, save_dir='buildAndTrainBirdsVsSquirrels_visualization_outputs'):
#     os.makedirs(save_dir, exist_ok=True)
    
#     plt.figure(figsize=(15, 5))
    
#     plt.subplot(1, 2, 1)
#     plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
#     plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
#     plt.title('Model Accuracy Over Time')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy')
#     plt.legend(loc='lower right')
#     plt.grid(True)

#     plt.subplot(1, 2, 2)
#     plt.plot(history.history['loss'], label='Training Loss', marker='o')
#     plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
#     plt.title('Model Loss Over Time')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
#     plt.legend(loc='upper right')
#     plt.grid(True)

#     plt.tight_layout()
#     plt.savefig(os.path.join(save_dir, 'buildAndTrainBirdsVsSquirrels_training_metrics.png'), dpi=300, bbox_inches='tight')
#     plt.close()

# def train_model(model, train_dataset, val_dataset, steps_per_epoch, validation_steps, epochs=50):
#     # Fixed: Use float instead of tf.Variable for initial learning rate
#     initial_learning_rate = 0.0002
    
#     optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
#     model.compile(
#         optimizer=optimizer,
#         loss='sparse_categorical_crossentropy',
#         metrics=['accuracy']
#     )

#     callbacks = [
#         WarmUpCosineDecayScheduler(
#             initial_lr=initial_learning_rate,  # Now passing float value
#             warmup_epochs=5,
#             total_epochs=epochs
#         ),
#         tf.keras.callbacks.EarlyStopping(
#             monitor='val_loss',
#             patience=20,
#             restore_best_weights=True,
#             min_delta=0.001
#         ),
#         tf.keras.callbacks.ModelCheckpoint(
#             'best_model.keras',
#             monitor='val_accuracy',
#             save_best_only=True,
#             mode='max',
#             verbose=1
#         )
#     ]
    
#     history = model.fit(
#         train_dataset,
#         validation_data=val_dataset,
#         epochs=epochs,
#         steps_per_epoch=steps_per_epoch,
#         validation_steps=validation_steps,
#         callbacks=callbacks,
#         verbose=1
#     )
    
#     return history

# def main():
#     try:
#         BATCH_SIZE = 32
#         EPOCHS = 50
#         NUM_CLASSES = 3

#         train_dataset, train_count = create_dataset('birds-vs-squirrels-train.tfrecords', BATCH_SIZE)
#         val_dataset, val_count = create_dataset('birds-vs-squirrels-validation.tfrecords', BATCH_SIZE, False)

#         steps_per_epoch = train_count // BATCH_SIZE
#         validation_steps = val_count // BATCH_SIZE

#         print(f"Steps per epoch: {steps_per_epoch}")
#         print(f"Validation steps: {validation_steps}")

#         model = create_model(NUM_CLASSES)
#         history = train_model(
#             model, 
#             train_dataset, 
#             val_dataset,
#             steps_per_epoch,
#             validation_steps,
#             EPOCHS
#         )

#         model.save('buildAndTrainBirdsVsSquirrels_final_model.keras')
#         plot_metrics(history)

#         print("\nTraining Results:")
#         for metric in ['accuracy', 'loss']:
#             train_val = history.history[metric][-1]
#             val_val = history.history[f'val_{metric}'][-1]
#             print(f"{metric}: {train_val:.4f} (val: {val_val:.4f})")

#         best_val_acc = max(history.history['val_accuracy'])
#         print(f"\nBest validation accuracy: {best_val_acc:.4f}")

#     except Exception as e:
#         print(f"An error occurred: {e}")
#         import traceback
#         traceback.print_exc()

# if __name__ == "__main__":
#     main()



import tensorflow as tf
import matplotlib.pyplot as plt
import json
import os
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix

class WarmUpCosineDecayScheduler(tf.keras.callbacks.Callback):
    def __init__(self, initial_lr, warmup_epochs, total_epochs):
        super(WarmUpCosineDecayScheduler, self).__init__()
        self.initial_lr = initial_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        
    def on_epoch_begin(self, epoch, logs=None):
        if epoch < self.warmup_epochs:
            lr = self.initial_lr * ((epoch + 1) / self.warmup_epochs)
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.initial_lr * 0.5 * (1 + np.cos(np.pi * progress))
        self.model.optimizer.learning_rate.assign(lr)

def create_dataset(file_path, batch_size=32, is_training=True):
    feature_description = {
        'image': tf.io.FixedLenFeature([], tf.string),
        'label': tf.io.FixedLenFeature([], tf.int64)
    }

    def _parse_function(example_proto):
        parsed_features = tf.io.parse_single_example(example_proto, feature_description)
        image = tf.io.decode_jpeg(parsed_features['image'], channels=3)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize_with_pad(image, 224, 224)
        image = image / 255.0
        label = parsed_features['label']
        return image, label

    def augment(image, label):
        if is_training:
            image = tf.image.random_flip_left_right(image)
            image = tf.image.random_flip_up_down(image)
            image = tf.image.random_brightness(image, 0.3)
            image = tf.image.random_contrast(image, 0.7, 1.3)
            image = tf.image.random_saturation(image, 0.7, 1.3)
            image = tf.image.random_hue(image, 0.2)
            image = tf.image.resize(image, [240, 240])
            image = tf.image.random_crop(image, [224, 224, 3])
            noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.01)
            image = image + noise
            image = tf.clip_by_value(image, 0.0, 1.0)
        return image, label

    try:
        dataset = tf.data.TFRecordDataset(file_path)
        count = sum(1 for _ in dataset)
        print(f"Found {count} examples in {file_path}")

        dataset = dataset.map(_parse_function, num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.map(augment, num_parallel_calls=tf.data.AUTOTUNE)

        if is_training:
            dataset = dataset.shuffle(1000)
            dataset = dataset.repeat()

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)

        return dataset, count

    except Exception as e:
        print(f"Error creating dataset from {file_path}: {e}")
        raise

def create_model(num_classes):
    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(224, 224, 3),
        include_top=False,
        weights='imagenet'
    )
    base_model.trainable = False

    model = tf.keras.Sequential([
        tf.keras.layers.Resizing(224, 224),
        base_model,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dense(512, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Activation('relu'),
        tf.keras.layers.Dropout(0.4),
        tf.keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    return model

def plot_metrics(history, save_dir='buiAndTrainBirdsVsSquirerels_visualization_outputs'):
    os.makedirs(save_dir, exist_ok=True)
    
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy', marker='o')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy', marker='s')
    plt.title('Model Accuracy Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss', marker='o')
    plt.plot(history.history['val_loss'], label='Validation Loss', marker='s')
    plt.title('Model Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'buiAndTrainBirdsVsSquirerels_training_metrics.png'), dpi=300, bbox_inches='tight')
    plt.close()

def train_model(model, train_dataset, val_dataset, steps_per_epoch, validation_steps, epochs=50):
    # Fixed: Use float instead of tf.Variable for initial learning rate
    initial_learning_rate = 0.0002
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate)
    
    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    callbacks = [
        WarmUpCosineDecayScheduler(
            initial_lr=initial_learning_rate,  # Now passing float value
            warmup_epochs=5,
            total_epochs=epochs
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=20,
            restore_best_weights=True,
            min_delta=0.001
        ),
        tf.keras.callbacks.ModelCheckpoint(
            'best_model.keras',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]
    
    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

def main():
    try:
        BATCH_SIZE = 32
        EPOCHS = 50
        NUM_CLASSES = 3

        train_dataset, train_count = create_dataset('birds-vs-squirrels-train.tfrecords', BATCH_SIZE)
        val_dataset, val_count = create_dataset('birds-vs-squirrels-validation.tfrecords', BATCH_SIZE, False)

        steps_per_epoch = train_count // BATCH_SIZE
        validation_steps = val_count // BATCH_SIZE

        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Validation steps: {validation_steps}")

        model = create_model(NUM_CLASSES)
        history = train_model(
            model, 
            train_dataset, 
            val_dataset,
            steps_per_epoch,
            validation_steps,
            EPOCHS
        )

        model.save('buiAndTrainBirdsVsSquirerels_final_model.keras')
        plot_metrics(history)

        print("\nTraining Results:")
        for metric in ['accuracy', 'loss']:
            train_val = history.history[metric][-1]
            val_val = history.history[f'val_{metric}'][-1]
            print(f"{metric}: {train_val:.4f} (val: {val_val:.4f})")

        best_val_acc = max(history.history['val_accuracy'])
        print(f"\nBest validation accuracy: {best_val_acc:.4f}")

    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()