## Vehicle Detection Using Image Processing: ##

## Introduction:
This project utilizes image processing for vehicle detection, crucial for traffic management and surveillance. We employ advanced algorithms to analyze images, identifying vehicles by shapes, colors, and movement.

## Setup and Prerequisites:
Google Colab: Utilize Google Colab for cloud-based execution.
Google Drive: Mount Google Drive for dataset access and storage (Link - https://drive.google.com/drive/folders/1jZHbE8eN8At5ZA-Jo7a6mgN7SlUVp-sG?usp=drive_link).

## Implementation:
Data Collection
Dataset: Kaggle Vehicle Dataset (Link- https://www.kaggle.com/code/l0new0lf/vehicle-classification/output).
Size: Approximately 2GB.

## Code Overview:
Implementation: Detailed code for ResNet, NasNet, and MobileNet models.
Training & Validation: Accuracy, confusion matrix, precision-recall curve.
Performance Metrics: Inference time, model size, AUC-ROC, scalability metrics.

## Model Development:
# ResNet Model: Pre-trained ResNet50 from ImageNet, customized for vehicle classification.
The ResNet model in this project utilizes a ResNet50 base from ImageNet, enhanced with custom layers like Global Average Pooling and a Dense layer for classification. This sequential model is adapted to process varying numbers of images per class. Key tools include TensorFlow, Keras, and Scikit-learn. The implementation covers data preprocessing using ImageDataGenerator and focuses on balanced class representation. The model's architecture, training, and validation processes aim to optimize accuracy, supported by visualizations such as confusion matrices and precision-recall curves. This approach ensures a comprehensive understanding of the model's performance and accuracy in image classification tasks.

## ResNet:
# Build the ResNet model
base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False
model = models.Sequential()
model.add(base_model)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(len(all_images_generator.class_indices), activation='softmax'))
# Compile the model
model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# NasNet & MobileNet Models: Utilization of TensorFlow, Keras, and Scikit-learn for model building and data management.
In this project, the NasNet and MobileNet models are developed using TensorFlow, Keras, and Scikit-learn. These models are built and managed by accessing datasets stored in Google Drive and entail preprocessing of images. Both models utilize pre-trained architectures, NASNetLarge and MobileNetV2, respectively. Key steps in their implementation include data loading, model configuration with frozen convolutional layers, and the addition of custom classification layers specific to vehicle detection. The models are trained with specific settings for batch size and epochs, and their performance is evaluated based on accuracy. This approach exemplifies the effective use of transfer learning in image classification.

## NasNet:
# Load pre-trained NASNetLarge model
base_model = NASNetLarge(weights='imagenet', include_top=False, input_shape=(331, 331, 3))
# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False
# Add custom classification layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(len(label_to_index), activation='softmax')(x)
model = Model(base_model.input, x)

## MobileNet:
# Load pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the convolutional layers
for layer in base_model.layers:
    layer.trainable = False
# Add custom classification layers
x = layers.GlobalAveragePooling2D()(base_model.output)
x = layers.Dense(512, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(len(label_to_index), activation='softmax')(x)
model = Model(base_model.input, x)

## Performance:
ResNet Accuracy improves with dataset size.
NasNet: 98% accuracy on a smaller dataset.
MobileNet: 93.75% accuracy.

## Conclusion:
- ResNet model's accuracy improves with larger datasets.
- NasNet achieved 98% accuracy on a smaller dataset.
- MobileNet reached 93.75% testing accuracy.
- Larger datasets posed compatibility challenges.
- Advanced GPUs are essential for processing large data efficiently.
- Models can achieve up to 80% accuracy before overfitting, even with increased data volumes.
