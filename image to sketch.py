import cv2
image = cv2.imread(r"C:\Users\Hammad\Downloads\GD.png")
cv2.imshow("GD", image)


# converting to grayscale
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("New GD", gray_image)


# converting grayscale to inverted image
inverted_image = 255 - gray_image
cv2.imshow("Inverted", inverted_image)
cv2.waitKey()

# blurring the image using gaussian function
blurred = cv2.GaussianBlur(inverted_image, (21, 21), 0)

# inverting the blurred image
inverted_blurred = 255 - blurred
pencil_sketch = cv2.divide(gray_image, inverted_blurred, scale=256.0)
cv2.imshow("Sketch", pencil_sketch)
cv2.waitKey(0)

# Comparing original with sketched
cv2.imshow("original", image)
cv2.imshow("sketch", pencil_sketch)
