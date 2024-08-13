from __future__ import division
import cv2

import numpy as np



def apply_mask(matrix, mask, fill_value):

    #print(flat[60])
    #print(flat[11940])
        
    masked = np.ma.array(matrix, mask=mask, fill_value=fill_value)
    # print('MASKED=',masked)
    return masked.filled()

def apply_threshold(matrix, low_value, high_value):
    low_mask = matrix < low_value
    matrix = apply_mask(matrix, low_mask, low_value)
    #print('Low MASK->',low_mask,'\nMatrix->',matrix)

    high_mask = matrix > high_value
    matrix = apply_mask(matrix, high_mask, high_value)

    return matrix


def simplest_cb(img, percent):
    # Check if the input image has at least one channel
    if img.ndim < 2:
        raise ValueError("Input image must have at least one channel")

    # Check if the percentage value is within the valid range
    if percent <= 0 or percent >= 100:
        raise ValueError("Percentage value must be between 0 and 100 (exclusive)")

    half_percent = percent / 200.0
    # print('HALF PERCENT->', half_percent)

    # Split the image into channels
    channels = cv2.split(img)
    num_channels = len(channels)
    # print('Number of channels:', num_channels)

    out_channels = []
    for channel in channels:
        # Check if the channel has the expected shape
        if len(channel.shape) != 2:
            raise ValueError("Channel shape is not valid")

        # Find the low and high percentile values (based on the input percentile)
        height, width = channel.shape
        vec_size = width * height
        flat = channel.reshape(vec_size)

        # Check if the flattened channel has the expected shape
        if len(flat.shape) != 1:
            raise ValueError("Flattened channel shape is not valid")

        flat = np.sort(flat)

        n_cols = flat.shape[0]

        low_val = flat[int(np.floor(n_cols * half_percent))]
        high_val = flat[int(np.ceil(n_cols * (1.0 - half_percent)))]

        # print("Lowval: ", low_val)
        # print("Highval: ", high_val)

        # Saturate below the low percentile and above the high percentile
        thresholded = apply_threshold(channel, low_val, high_val)

        # Scale the channel
        normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
        out_channels.append(normalized)

    # Merge the processed channels back into a single image
    processed_img = cv2.merge(out_channels)
    return processed_img


# def simplest_cb(img, percent):
#     assert percent > 0 and percent < 100
#     half_percent = percent / 200.0

#     if img.ndim == 2:
#         img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
#     elif img.shape[2] == 3:
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
#     else:
#         print("Invalid image format")
#         return None

#     channels = cv2.split(img)

#     out_channels = []
#     for channel in channels[:2]:  # Process only the first two channels (Y and Cr)
#         assert len(channel.shape) == 2
#         height, width = channel.shape
#         vec_size = width * height
#         flat = channel.reshape(vec_size)
#         assert len(flat.shape) == 1

#         flat = np.sort(flat)

#         n_cols = flat.shape[0]

#         low_val = flat[math.floor(n_cols * half_percent)]
#         high_val = flat[math.ceil(n_cols * (1.0 - half_percent))]

#         thresholded = apply_threshold(channel, low_val, high_val)
#         normalized = cv2.normalize(thresholded, thresholded.copy(), 0, 255, cv2.NORM_MINMAX)
#         out_channels.append(normalized)

#     out_channels.append(channels[2])  # Append the Cb channel without processing

#     out = cv2.merge(out_channels)
#     if img.shape[2] == 3:
#         out = cv2.cvtColor(out, cv2.COLOR_YCrCb2BGR)

#     return out




if __name__ == '__main__':
    #img = cv2.imread(sys.argv[1])
    file_path = r'codes\forest.jpg'
    
    print(f"File Path: {file_path}")
    img = cv2.imread(file_path)

    out = simplest_cb(img, 1)
    cv2.imshow("Before", img)
    cv2.imshow("After", out)
    cv2.waitKey(0)
