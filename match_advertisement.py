import cv2
import numpy as np
from skimage.color import rgb2lab, deltaE_cie76

# Function to replace pixels in the frame and reference frame within a specified color difference with pixels from the rectangular-shaped advertisement
def replace_matching_pixels_with_color_difference(frame, reference_frame, rectangular_advertisement, x, y, color_difference_threshold=60, dilation_kernel_size=7, erosion_kernel_size=5, shadow_intensity=0.5):
    h, w = rectangular_advertisement.shape[0], rectangular_advertisement.shape[1]

    # Extract the region where the rectangular advertisement will be placed in the current frame
    region_in_frame = frame[y:y + h, x:x + w]

    # Extract the corresponding region from the reference frame
    region_in_reference = reference_frame[y:y + h, x:x + w]

    # Convert regions to Lab color space
    lab_frame = rgb2lab(region_in_frame)
    lab_reference = rgb2lab(region_in_reference)

    # Calculate color difference using CIE76 formula
    color_difference = deltaE_cie76(lab_frame, lab_reference)

    # Create a mask based on color difference
    mask = (color_difference <= color_difference_threshold).astype(np.uint8)

    # Apply morphological operations (dilation and erosion) for clarity
    kernel_dilation = np.ones((dilation_kernel_size, dilation_kernel_size), np.uint8)
    kernel_erosion = np.ones((erosion_kernel_size, erosion_kernel_size), np.uint8)
    mask = cv2.dilate(mask, kernel_dilation, iterations=1)
    mask = cv2.erode(mask, kernel_erosion, iterations=1)

    # Calculate shadow mask
    shadow_mask = (mask > 0).astype(np.uint8)

    # Apply shadow intensity to the shadow mask
    shadow_mask = (shadow_mask * shadow_intensity).astype(np.uint8)

    # Warp the rectangular-shaped advertisement using perspective transformation
    rectangular_advertisement_warped, mask_warped = warp_rectangle(rectangular_advertisement, mask)

    # Create a shadow using the shadow mask
    shadow = np.zeros_like(frame)
    shadow[y:y + h, x:x + w] = rectangular_advertisement_warped * shadow_mask[:, :, np.newaxis]

    # Add the shadow to the frame
    frame_with_shadow = frame + shadow

    # Replace pixels in the frame with pixels from the rectangular-shaped advertisement based on the enhanced mask
    frame_with_shadow[y:y + h, x:x + w] = rectangular_advertisement_warped * mask_warped[:, :, np.newaxis] + frame[y:y + h, x:x + w] * (1 - mask_warped[:, :, np.newaxis])

    return frame_with_shadow, reference_frame

# Function to warp an image into a rectangular shape
def warp_rectangle(image, mask):
    # Define the four corners of the rectangle in the source image
    src_points = np.array([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]], dtype=np.float32)
    
    # Define the four corners of the rectangle in the destination image (desired rectangular shape)
    dst_points = np.array([[0, 0], [image.shape[1] - 1, 0], [0, image.shape[0] - 1], [image.shape[1] - 1, image.shape[0] - 1]], dtype=np.float32)

    # Compute the perspective transformation matrix
    perspective_matrix = cv2.getPerspectiveTransform(src_points, dst_points)

    # Apply the perspective transformation to the image
    rectangular_warped = cv2.warpPerspective(image, perspective_matrix, (image.shape[1], image.shape[0]))
    mask_warped = cv2.warpPerspective(mask, perspective_matrix, (image.shape[1], image.shape[0]))

    return rectangular_warped, mask_warped

# Rest of the code remains unchanged...

# Load video
cap = cv2.VideoCapture('Untitled video - Made with Clipchamp (1).mp4')

# Load original advertisement image
original_advertisement = cv2.imread('klipartz.com.png')

# Load reference frame (ground without any player)
reference_frame = cv2.imread('reference_frame.png')  # Replace with the path to your reference frame









# Manually specify pixel coordinates for placing the rectangular-shaped advertisement
advertisement_x = 1200
advertisement_y = 260

# Parameters for the function
color_difference_threshold = 20
dilation_kernel_size = 5
erosion_kernel_size = 3
shadow_intensity = 0.5  # Adjust the shadow intensity as needed

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Ensure the rectangular-shaped advertisement fits within the frame
    if advertisement_x + original_advertisement.shape[1] <= frame.shape[1] and advertisement_y + original_advertisement.shape[0] <= frame.shape[0]:
        # Replace pixels within a specified color difference with pixels from the rectangular-shaped advertisement
        frame_with_shadow, reference_frame = replace_matching_pixels_with_color_difference(
            frame, reference_frame, original_advertisement, advertisement_x, advertisement_y,
            color_difference_threshold=color_difference_threshold,
            dilation_kernel_size=dilation_kernel_size,
            erosion_kernel_size=erosion_kernel_size,
            shadow_intensity=shadow_intensity
        )
        # cv2.rectangle(frame, (advertisement_x, advertisement_y), (advertisement_x + original_advertisement.shape[1], advertisement_y + original_advertisement.shape[0]), (0, 255, 0), 2)
        # cv2.rectangle(reference_frame, (advertisement_x, advertisement_y), (advertisement_x + original_advertisement.shape[1], advertisement_y + original_advertisement.shape[0]), (0, 255, 0), 2)

    cv2.imshow('Virtual Advertisement', frame_with_shadow)
    # cv2.imshow('Reference Frame with Advertisement', reference_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
