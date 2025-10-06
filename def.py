import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import math
import os

def get_geotagging(exif_data):
    """Extract GPS information from EXIF data"""
    if not exif_data:
        return None
    
    gps_info = {}
    for key, val in exif_data.items():
        if key == 'GPSInfo':
            for t in val:
                sub_tag = GPSTAGS.get(t, t)
                gps_info[sub_tag] = val[t]
    
    if not gps_info:
        return None
    
    return gps_info

def dms_to_decimal(dms, ref):
    """Convert DMS (Degrees, Minutes, Seconds) to decimal degrees"""
    degrees = dms[0]
    minutes = dms[1] / 60.0
    seconds = dms[2] / 3600.0
    
    decimal = degrees + minutes + seconds
    
    if ref in ['S', 'W']:
        decimal = -decimal
    
    return decimal

def get_decimal_coordinates(geotags):
    """Get decimal latitude and longitude from GPS tags"""
    if not geotags:
        return None, None
    
    latitude = dms_to_decimal(geotags['GPSLatitude'], geotags['GPSLatitudeRef'])
    longitude = dms_to_decimal(geotags['GPSLongitude'], geotags['GPSLongitudeRef'])
    
    return latitude, longitude

def extract_exif_data(image_path):
    """Extract EXIF data from an image"""
    try:
        image = Image.open(image_path)
        exif_data = image._getexif()
        
        if not exif_data:
            return None
        
        # Create a dictionary of readable EXIF tags
        exif = {}
        for tag, value in exif_data.items():
            decoded = TAGS.get(tag, tag)
            exif[decoded] = value
        
        return exif
    except Exception as e:
        print(f"Error extracting EXIF data: {e}")
        return None

def calculate_ndvi(image_path):
    """Calculate NDVI (Normalized Difference Vegetation Index) for an image"""
    try:
        # Read the image
        img = cv2.imread(image_path)
        
        if img is None:
            raise ValueError(f"Could not read image at {image_path}")
        
        # Convert to RGB (OpenCV uses BGR by default)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Convert to float32 for calculations
        img_float = img_rgb.astype(np.float32) / 255.0
        
        # Extract red and near-infrared bands
        # For a standard RGB image, we'll use the red channel and assume NIR is similar to green
        # Note: For accurate NDVI, you'd need actual NIR data from a multispectral sensor
        red = img_float[:, :, 0]
        nir = img_float[:, :, 1]  # This is a simplification - real NDVI requires NIR band
        
        # Calculate NDVI
        ndvi = (nir - red) / (nir + red + 1e-10)  # Add small value to avoid division by zero
        
        return ndvi
    except Exception as e:
        print(f"Error calculating NDVI: {e}")
        return None

def align_images(image1_path, image2_path):
    """Align two images using feature matching"""
    try:
        # Read images
        img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            raise ValueError("Could not read one or both images")
        
        # Initialize ORB detector
        orb = cv2.ORB_create(1000)
        
        # Find keypoints and descriptors
        kp1, des1 = orb.detectAndCompute(img1, None)
        kp2, des2 = orb.detectAndCompute(img2, None)
        
        # Create BFMatcher object
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        
        # Match descriptors
        matches = bf.match(des1, des2)
        
        # Sort matches by distance
        matches = sorted(matches, key=lambda x: x.distance)
        
        # Extract location of good matches
        points1 = np.zeros((len(matches), 2), dtype=np.float32)
        points2 = np.zeros((len(matches), 2), dtype=np.float32)
        
        for i, match in enumerate(matches):
            points1[i, :] = kp1[match.queryIdx].pt
            points2[i, :] = kp2[match.trainIdx].pt
        
        # Find homography
        h, mask = cv2.findHomography(points1, points2, cv2.RANSAC)
        
        # Use homography to align image2 to image1
        height, width = img1.shape
        img2_aligned = cv2.warpPerspective(img2, h, (width, height))
        
        return img2_aligned
    except Exception as e:
        print(f"Error aligning images: {e}")
        return None

def detect_deforestation(image1_path, image2_path, threshold=0.2):
    """
    Detect deforestation by comparing two images
    
    Parameters:
    - image1_path: Path to the earlier image
    - image2_path: Path to the later image
    - threshold: Threshold for significant vegetation change (0-1)
    
    Returns:
    - result_image: Image highlighting deforested areas
    - deforestation_percentage: Percentage of area deforested
    - coordinates: GPS coordinates of the images
    """
    try:
        # Extract GPS coordinates from both images
        exif1 = extract_exif_data(image1_path)
        exif2 = extract_exif_data(image2_path)
        
        geotags1 = get_geotagging(exif1) if exif1 else None
        geotags2 = get_geotagging(exif2) if exif2 else None
        
        lat1, lon1 = get_decimal_coordinates(geotags1) if geotags1 else (None, None)
        lat2, lon2 = get_decimal_coordinates(geotags2) if geotags2 else (None, None)
        
        coordinates = {
            "image1": {"latitude": lat1, "longitude": lon1},
            "image2": {"latitude": lat2, "longitude": lon2}
        }
        
        # Align images if they're from slightly different perspectives
        img2_aligned = align_images(image1_path, image2_path)
        
        if img2_aligned is None:
            # If alignment fails, use original image
            img2 = cv2.imread(image2_path)
            img2_aligned = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
        
        # Calculate NDVI for both images
        ndvi1 = calculate_ndvi(image1_path)
        ndvi2 = calculate_ndvi(image2_path)
        
        if ndvi1 is None or ndvi2 is None:
            raise ValueError("Could not calculate NDVI for one or both images")
        
        # Calculate NDVI difference
        ndvi_diff = ndvi1 - ndvi2
        
        # Threshold the difference to find significant vegetation loss
        deforestation_mask = ndvi_diff > threshold
        
        # Calculate percentage of deforested area
        deforestation_percentage = np.sum(deforestation_mask) / deforestation_mask.size * 100
        
        # Create result visualization
        result_image = np.zeros((ndvi1.shape[0], ndvi1.shape[1], 3), dtype=np.uint8)
        
        # Green: areas with no significant change
        # Red: areas with significant vegetation loss (deforestation)
        result_image[~deforestation_mask] = [0, 255, 0]  # Green
        result_image[deforestation_mask] = [0, 0, 255]   # Red
        
        # Overlay on original image
        original_img = cv2.imread(image1_path)
        original_img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
        
        # Blend the result with the original image
        alpha = 0.5  # Transparency factor
        result_overlay = cv2.addWeighted(original_img_rgb, alpha, result_image, 1-alpha, 0)
        
        return result_overlay, deforestation_percentage, coordinates
    except Exception as e:
        print(f"Error in deforestation detection: {e}")
        return None, None, None

def main():
    # Paths to the two images
    image1_path = input("Enter path to the first image (earlier time period): ")
    image2_path = input("Enter path to the second image (later time period): ")
    
    # Check if files exist
    if not os.path.exists(image1_path) or not os.path.exists(image2_path):
        print("One or both image files do not exist.")
        return
    
    # Detect deforestation
    result_image, deforestation_percentage, coordinates = detect_deforestation(image1_path, image2_path)
    
    if result_image is None:
        print("Failed to analyze images for deforestation.")
        return
    
    # Display results
    plt.figure(figsize=(12, 8))
    plt.imshow(result_image)
    plt.title(f"Deforestation Detection - {deforestation_percentage:.2f}% of area affected")
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    
    # Print GPS coordinates if available
    if coordinates["image1"]["latitude"] is not None:
        print(f"\nImage 1 GPS Coordinates: {coordinates['image1']['latitude']:.6f}, {coordinates['image1']['longitude']:.6f}")
    if coordinates["image2"]["latitude"] is not None:
        print(f"Image 2 GPS Coordinates: {coordinates['image2']['latitude']:.6f}, {coordinates['image2']['longitude']:.6f}")
    
    # Interpret results
    print("\nDeforestation Analysis Results:")
    print(f"- Percentage of area showing vegetation loss: {deforestation_percentage:.2f}%")
    
    if deforestation_percentage < 5:
        print("- Conclusion: Minimal to no deforestation detected")
    elif deforestation_percentage < 15:
        print("- Conclusion: Low level of deforestation detected")
    elif deforestation_percentage < 30:
        print("- Conclusion: Moderate deforestation detected")
    else:
        print("- Conclusion: Significant deforestation detected")
    
    # Save the result image
    output_path = "deforestation_result.png"
    cv2.imwrite(output_path, cv2.cvtColor(result_image, cv2.COLOR_RGB2BGR))
    print(f"\nResult image saved as: {output_path}")

if __name__ == "__main__":
    main()