import os
import sys

import numpy as np
import torch
from PIL import Image
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import ExplicitVRLittleEndian

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from core.models.factory import ModelFactory


def main():
    fixtures_dir = os.path.dirname(__file__)
    os.makedirs(fixtures_dir, exist_ok=True)

    # 1. Generate dummy_xray.png
    png_path = os.path.join(fixtures_dir, "dummy_xray.png")
    img = Image.new("RGB", (224, 224), color="gray")
    img.save(png_path)
    print(f"Generated {png_path}")

    # 2. Generate dummy_dicom.dcm
    dcm_path = os.path.join(fixtures_dir, "dummy_dicom.dcm")
    
    # Create a minimal DICOM dataset structure
    file_meta = Dataset()
    file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.1.1" # Digital X-Ray Image Storage - For Presentation
    file_meta.MediaStorageSOPInstanceUID = "1.2.3.4.5.6.7"
    file_meta.ImplementationClassUID = "1.2.3.4.5.6"
    file_meta.TransferSyntaxUID = ExplicitVRLittleEndian
    
    ds = FileDataset(dcm_path, {}, file_meta=file_meta, preamble=b"\x00" * 128)
    ds.PatientName = "Test^Patient"
    ds.PatientID = "123456"
    ds.Modality = "DX"
    ds.Rows = 224
    ds.Columns = 224
    ds.BitsAllocated = 16
    ds.BitsStored = 12
    ds.HighBit = 11
    ds.PixelRepresentation = 0
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    
    # Generate pixel array
    pixel_array = np.random.randint(0, 4095, (224, 224), dtype=np.uint16)
    ds.PixelData = pixel_array.tobytes()
    
    # Save the file
    ds.is_little_endian = True
    ds.is_implicit_VR = False
    ds.save_as(dcm_path)
    print(f"Generated {dcm_path}")

    # 3. Generate mini_tier1_weights.pth
    t1_path = os.path.join(fixtures_dir, "mini_tier1_weights.pth")
    t1_model = ModelFactory.create("mobilenet_v2", num_classes=2, pretrained=False)
    torch.save(t1_model.state_dict(), t1_path)
    print(f"Generated {t1_path}")

    # 4. Generate mini_tier2_weights.pth
    t2_path = os.path.join(fixtures_dir, "mini_tier2_weights.pth")
    t2_model = ModelFactory.create("efficientnet_b4", num_classes=2, pretrained=False)
    torch.save(t2_model.state_dict(), t2_path)
    print(f"Generated {t2_path}")

if __name__ == "__main__":
    main()
