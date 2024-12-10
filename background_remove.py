import numpy as np
import cv2


def remove_background(image_path, output_path):
  """Removes the background from an image.

  Args:
    image_path: Path to the input image.
    output_path: Path to save the output image.
  """

  # Read the image
  img = cv2.imread(image_path)

  # Convert to HSV color space
  hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

  # Define the color range for the background
  # Adjust these values based on your specific image
  lower_bound = np.array([0, 0, 0])  # Lower HSV values for background
  upper_bound = np.array([255, 255, 255])  # Upper HSV values for background

  # Create a mask for the background
  mask = cv2.inRange(hsv, lower_bound, upper_bound)

  # Invert the mask to get the foreground
  mask_inv = cv2.bitwise_not(mask)

  # Apply the mask to the image
  bg_removed = cv2.bitwise_and(img, img, mask=mask_inv)

  # Create a transparent background
  bg_transparent = np.zeros_like(img, np.uint8)
  bg_transparent = cv2.cvtColor(bg_transparent, cv2.COLOR_BGR2BGRA) 
  bg_transparent[:, :, 3] = 0 # Set alpha channel to 0 for transparency

  bg_removed = cv2.resize(bg_removed, (bg_transparent.shape[1], bg_transparent.shape[0]))
  bg_removed = cv2.cvtColor(bg_removed, cv2.COLOR_BGR2BGRA)

  # Combine the foreground with the transparent background
  final_image = cv2.addWeighted(bg_transparent, 1, bg_removed, 1, 0)

  # Save the output image
  cv2.imwrite(output_path, final_image)

# Example usage:
image_path = "images/testface.jpg"
output_path = "processed_images/output_image.png"
remove_background(image_path, output_path)


