library(keras)

# Load Fashion MNIST dataset
data <- dataset_fashion_mnist()
x_train <- data$train$x / 255
y_train <- data$train$y
x_test <- data$test$x / 255
y_test <- data$test$y

# Reshape images for CNN input
x_train <- array_reshape(x_train, c(nrow(x_train), 28, 28, 1))
x_test <- array_reshape(x_test, c(nrow(x_test), 28, 28, 1))

# Define CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3,3), activation = 'relu', input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3,3), activation = 'relu') %>%
  layer_max_pooling_2d(pool_size = c(2,2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3,3), activation = 'relu') %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dense(units = 10, activation = 'softmax')

# Compile model
model %>% compile(
  optimizer = 'adam',
  loss = 'sparse_categorical_crossentropy',
  metrics = 'accuracy'
)

# Train model
model %>% fit(x_train, y_train, epochs = 10, validation_data = list(x_test, y_test))

# Make predictions on two test images
sample_images <- x_test[1:2,,, drop=FALSE]
predictions <- model %>% predict(sample_images)
predicted_classes <- apply(predictions, 1, which.max) - 1

# Display predictions
par(mfrow=c(1,2))
for (i in 1:2) {
  image(matrix(x_test[i,,], nrow=28, ncol=28), col=gray.colors(255))
  title(paste('Predicted:', predicted_classes[i]))
}

# Save model
save_model_hdf5(model, "fashion_mnist_cnn_r.h5")
