import cv2  # OpenCV for image processing operations
import numpy as np  # NumPy for numerical computations

# Step 1: Preprocess the image
def preprocess_image(image):
    # Convert image from BGR (color) to Grayscale for simpler analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian Blur to reduce noise and smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Use Otsu's method for automatic thresholding
    # THRESH_BINARY_INV: Inverts binary image so signature becomes white, background black
    _, binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Return the clean, binary version of the image
    return binary

# Step 2: Extract basic features (contour area and perimeter)
def extract_features(image):
    # Detect edges using the Canny edge detector
    edges = cv2.Canny(image, 50, 150)

    # Find contours based on edges; only external contours are considered
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Sum area and perimeter of all contours detected
    total_area = sum([cv2.contourArea(c) for c in contours])
    total_perimeter = sum([cv2.arcLength(c, True) for c in contours])

    # Return as a NumPy array for comparison
    return np.array([total_area, total_perimeter])

# Step 3: Compare two feature vectors
def compare_signatures(feat1, feat2):
    # Calculate Euclidean distance between the two feature vectors
    # Smaller distance = more similar signatures
    distance = np.linalg.norm(feat1 - feat2)
    return distance

# Step 4: Extract more advanced shape features from the image
def extract_advanced_features(image):
    # Find contours in the preprocessed (binary) image
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # If no contours are found, return default values
    if not contours:
        return 0, 0, 0, np.zeros(7)

    # Select the largest contour (assumed to be the main signature)
    largest_contour = max(contours, key=cv2.contourArea)

    # Compute the area and perimeter of the largest contour
    area = cv2.contourArea(largest_contour)
    perimeter = cv2.arcLength(largest_contour, True)

    # Compute the bounding rectangle and calculate aspect ratio (width / height)
    x, y, w, h = cv2.boundingRect(largest_contour)
    aspect_ratio = float(w) / h if h != 0 else 0

    # Calculate image moments (statistical properties of the shape)
    moments = cv2.moments(largest_contour)

    # Convert the moments into 7 Hu Moments (invariant to scale, rotation, and translation)
    hu_moments = cv2.HuMoments(moments).flatten()

    # Apply log scale to Hu Moments for stability and normalization
    # Keeps sign to capture shape characteristics properly
    hu_moments = -np.sign(hu_moments) * np.log10(np.abs(hu_moments) + 1e-10)

    # Return all advanced shape descriptors
    return area, perimeter, aspect_ratio, hu_moments
