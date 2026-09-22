import numpy as np
import SimpleITK as sitk
from radiomics import featureextractor
import logging

# Suppress verbose pyradiomics logging during batched extraction
logging.getLogger("radiomics").setLevel(logging.ERROR)

class RadiomicsExtractor:
    """
    Extracts quantitative tissue properties from the CNN-defined ROI.
    """
    def __init__(self):
        # Initialize the feature extractor with settings optimized for 2D MRI slices
        self.extractor = featureextractor.RadiomicsFeatureExtractor()
        
        # Configure the extractor to strictly evaluate 2D features
        self.extractor.settings['force2D'] = True
        
        # Enable the required feature families[cite: 2, 3]
        self.extractor.enableFeatureClassByName('firstorder')
        self.extractor.enableFeatureClassByName('glcm')
        self.extractor.enableFeatureClassByName('glrlm')
        self.extractor.enableFeatureClassByName('glszm')
        self.extractor.enableFeatureClassByName('ngtdm')
        
        # Enable multi-resolution wavelet filters to capture spatial frequencies[cite: 2, 3]
        self.extractor.enableImageTypeByName('Wavelet')

    def extract_features(self, image_array: np.ndarray, mask_array: np.ndarray) -> dict:
        """
        Takes a 2D image array and binary mask array, returning a dictionary of radiomics features.
        """
        # Ensure mask is strictly binary (0 or 1) and matches image dimensions
        mask_array = (mask_array > 0).astype(np.uint8)
        
        if image_array.shape != mask_array.shape:
            raise ValueError("Image and mask arrays must have identical dimensions.")
            
        # Convert numpy arrays to SimpleITK Image objects required by pyradiomics
        sitk_image = sitk.GetImageFromArray(image_array)
        sitk_mask = sitk.GetImageFromArray(mask_array)
        
        # Execute extraction
        try:
            result = self.extractor.execute(sitk_image, sitk_mask)
            
            # Filter out diagnostic metadata keys, keeping only the computed numerical features
            feature_dict = {key: val.item() for key, val in result.items() if not key.startswith('diagnostics_')}
            return feature_dict
            
        except Exception as e:
            print(f"Extraction failed: {e}")
            return {}

# Example execution:
# radiomics_engine = RadiomicsExtractor()
# features = radiomics_engine.extract_features(mri_slice_numpy, roi_mask_numpy)