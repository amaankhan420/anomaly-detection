import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from skimage.feature.peak import peak_local_max
from skimage.transform import resize
from keras.models import Model
from keras.models import load_model
import scipy

model = load_model("C:/Users/user/OneDrive/Desktop/project/model_training/my_model.h5")


def plot_heat_map(img):
    img = resize(img, (224, 224, 3))
    pred = model.predict(np.expand_dims(img, axis=0))
    pred_class = np.argmax(pred)

    # Check if the predicted class is 1, indicating no anomalies found
    if pred_class == 1:
        plt.imshow(img)
        plt.title("No anomalies found")
        plt.axis('off')
        plt.show()
        return

    # Get weights for all classes from the prediction layer
    last_layer_weights = model.layers[-1].get_weights()[0]  # Prediction layer
    # Get weights for the predicted class.
    last_layer_weights_for_pred = last_layer_weights[:, pred_class]
    # Get output from the last conv. layer
    last_conv_model = Model(model.input, model.get_layer("block5_conv3").output)
    last_conv_output = last_conv_model.predict(img[np.newaxis, :, :, :])
    last_conv_output = np.squeeze(last_conv_output)
    # Upsample/resize the last conv. output to the same size as the original image
    h = int(img.shape[0] / last_conv_output.shape[0])
    w = int(img.shape[1] / last_conv_output.shape[1])
    upsampled_last_conv_output = scipy.ndimage.zoom(last_conv_output, (h, w, 1), order=1)
    heat_map = np.dot(upsampled_last_conv_output.reshape((img.shape[0] * img.shape[1], 512)),
                      last_layer_weights_for_pred).reshape(img.shape[0], img.shape[1])

    # Detect peaks (hot spots) in the heat map. We will set it to detect maximum 5 peaks.
    # with rel threshold of 0.5 (compared to the max peak).
    peak_coords = peak_local_max(heat_map, num_peaks=5, threshold_rel=0.5, min_distance=10)

    # Create subplot with 1 row and 2 columns
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot original image
    axs[0].imshow(img.astype('float32').reshape(img.shape[0], img.shape[1], 3))
    axs[0].set_title('Original Image')

    # Plot heat map
    axs[1].imshow(img.astype('float32').reshape(img.shape[0], img.shape[1], 3))
    axs[1].imshow(heat_map, cmap='jet', alpha=0.30)
    for i in range(0, peak_coords.shape[0]):
        y = peak_coords[i, 0]
        x = peak_coords[i, 1]
        axs[1].add_patch(Rectangle((x - 25, y - 25), 70, 70, linewidth=1, edgecolor='r', facecolor='none'))
        # Overlay rectangles on the original image
        axs[0].add_patch(Rectangle((x - 25, y - 25), 70, 70, linewidth=1, edgecolor='r', facecolor='none'))
    axs[1].set_title('Heat Map')

    plt.show()
