import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import dicom
import os
import scipy.ndimage
import matplotlib.pyplot as plt
import math

from skimage import measure, morphology
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# Some constants 
MIN_BOUND = -1000.0
MAX_BOUND = 400.0


def preprocess_patient(path):
    # Get pixels
    patient_scan = load_scan(path)
    patient_pixels = get_pixels_hu(patient_scan)
    
    # Resample and save new spacing
    pix_resampled, spacing = resample(patient_pixels, patient_scan, [1,1,1])

    # Segment lungs to produce a mask and dilate the mask
    segmentedLungMask = segment_lung_mask(pix_resampled, True)
    if np.mean(segmentedLungMask >= .1):
        rawMask = segmentedLungMask # if >=10% of image is retained with segmentation, let's presume the mask worked
        dilatedMask = morphology.binary_dilation(image=rawMask) # <- EDIT THIS TO ALTER VERTICAL SPREADING OF MASK
    else:
        dilatedMask = np.ones(segmentedLungMask.shape, dtype=np.int8) # fall back to not masking anything if the segmentation isolated only a small portion
    
    
    # Normalize the pixels in the image and then mask them with the mask
    normalizedPixels = normalize(pix_resampled)
    maskedPixels = normalizedPixels*dilatedMask
    
    
    scaleDown = True
    
    if (scaleDown):
        # Now scale to half the size
        scaledPixels = scipy.ndimage.interpolation.zoom(maskedPixels, 0.5, mode='nearest')
    else:
        # Do no scaling, we'll just crop to 128
        scaledPixels = maskedPixels
  
    # Standardize the volume
    croppedPixels = standardize_volume(scaledPixels, 128, 0.0)
    

    
    # Return the masked pixels, plus the spacing
    return croppedPixels



# Load the scans in given folder path
def load_scan(path):
    slices = [dicom.read_file(path + '/' + s) for s in os.listdir(path)]
    slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
    try:
        slice_thickness = np.abs(slices[0].ImagePositionPatient[2] - slices[1].ImagePositionPatient[2])
    except:
        slice_thickness = np.abs(slices[0].SliceLocation - slices[1].SliceLocation)
        
    for s in slices:
        s.SliceThickness = slice_thickness
        
    return slices


def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)


def resample(image, scan, new_spacing=[1,1,1]):
    # Determine current pixel spacing
    spacing = map(float, ([scan[0].SliceThickness] + scan[0].PixelSpacing))
    spacing = np.array(list(spacing))

    resize_factor = spacing / new_spacing
    new_real_shape = image.shape * resize_factor
    new_shape = np.round(new_real_shape)
    real_resize_factor = new_shape / image.shape
    new_spacing = spacing / real_resize_factor
    
    image = scipy.ndimage.interpolation.zoom(image, real_resize_factor, mode='nearest')
    
    return image, new_spacing

def largest_label_volume(im, bg=-1):
    vals, counts = np.unique(im, return_counts=True)

    counts = counts[vals != bg]
    vals = vals[vals != bg]

    if len(counts) > 0:
        return vals[np.argmax(counts)]
    else:
        return None

def segment_lung_mask(image, fill_lung_structures=True):
    
    # not actually binary, but 1 and 2. 
    # 0 is treated as background, which we do not want
    binary_image = np.array(image > -320, dtype=np.int8)+1
    labels = measure.label(binary_image)
    
    # Pick the pixel in the very corner to determine which label is air.
    #   Improvement: Pick multiple background labels from around the patient
    #   More resistant to "trays" on which the patient lays cutting the air 
    #   around the person in half
    background_label = labels[0,0,0]
    
    #Fill the air around the person
    binary_image[background_label == labels] = 2
    
    
    # Method of filling the lung structures (that is superior to something like 
    # morphological closing)
    if fill_lung_structures:
        # For every slice we determine the largest solid structure
        for i, axial_slice in enumerate(binary_image):
            axial_slice = axial_slice - 1
            labeling = measure.label(axial_slice)
            l_max = largest_label_volume(labeling, bg=0)
            
            if l_max is not None: #This slice contains some lung
                binary_image[i][labeling != l_max] = 1

    
    binary_image -= 1 #Make the image actual binary
    binary_image = 1-binary_image # Invert it, lungs are now 1
    
    # Remove other air pockets insided body
    labels = measure.label(binary_image, background=0)
    l_max = largest_label_volume(labels, bg=0)
    if l_max is not None: # There are air pockets
        binary_image[labels != l_max] = 0
 
    return binary_image
    
def normalize(image):
    image = (image - MIN_BOUND) / (MAX_BOUND - MIN_BOUND)
    image[image>1] = 1.
    image[image<0] = 0.
    return image


#####
# Standardize volumes by appropriately cropping too-large dimensions, and filling in too-small dimensions
#####
def standardize_volume(pixels, desiredPixelsCount=256, fillValue=0.0):
    
    # Crop down each dimension to maximum of desired size, if needed.

    existingDims = pixels.shape    
    newPixelRanges = []
    
    for existingDimLength in existingDims:
        if existingDimLength > desiredPixelsCount:
            pixelsToCut = existingDimLength - desiredPixelsCount
            leftPixelsToCut = int(math.floor(pixelsToCut/2.0))
            newPixelRange = tuple([leftPixelsToCut,leftPixelsToCut+desiredPixelsCount])
        else:
            newPixelRange = tuple([0,0+existingDimLength])
        newPixelRanges.append(newPixelRange)
        
        
    newPixels = pixels.copy()
    newPixels = newPixels[newPixelRanges[0][0]:newPixelRanges[0][1],
                          newPixelRanges[1][0]:newPixelRanges[1][1],
                          newPixelRanges[2][0]:newPixelRanges[2][1]]     
    # Now fill in any extra space needed if dimensions are short
    revisedDims = newPixels.shape
    
    #print(revisedDims)
    
    additionalPixelCounts = []
    
    for existingDimLength in revisedDims:
        if existingDimLength < desiredPixelsCount:
            additionalPixelsTotal = desiredPixelsCount - existingDimLength
            additionalPixelsLeft = int(math.floor(additionalPixelsTotal/2.0))
            additionalPixelsRight = additionalPixelsTotal - additionalPixelsLeft
            additionalPixelCounts.append(tuple([additionalPixelsLeft,additionalPixelsRight]))
        else:
            additionalPixelCounts.append(tuple([0,0]))
            
    #print(additionalPixelCounts)
            
    
    for dimNum, additionalPixelCountTup in enumerate(additionalPixelCounts):
        #print(dimNum)
        # Add pixels to the beginning of this dimension (if necessary)
        workingDims = newPixels.shape
        if( additionalPixelCountTup[0] > 0 ):
            pixelAdditionsLeftDims = [workingDims[0],workingDims[1],workingDims[2]]
            pixelAdditionsLeftDims[dimNum] = additionalPixelCountTup[0]
            pixelAdditionsLeft = np.ones(pixelAdditionsLeftDims)
            pixelAdditionsLeft.fill(fillValue)
            newPixels = np.concatenate([pixelAdditionsLeft,newPixels], axis=dimNum)
        
        # Add pixels to the end of this dimension (if necessary)
        workingDims = newPixels.shape
        if( additionalPixelCountTup[1] > 0 ):
            pixelAdditionsRightDims =  [workingDims[0],workingDims[1],workingDims[2]]
            pixelAdditionsRightDims[dimNum] = additionalPixelCountTup[1]
            pixelAdditionsRight = np.ones(pixelAdditionsRightDims)
            pixelAdditionsRight.fill(fillValue)
            newPixels = np.concatenate([newPixels, pixelAdditionsRight], axis=dimNum)
        
        
        #print(newPixels.shape)
            
        
    return(newPixels)

#####
# End standardize dimensions
#####
