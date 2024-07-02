import numpy as np
import matplotlib.pyplot as plt
import pydicom
from pydicom.dataset import Dataset, FileDataset
from pydicom.uid import generate_uid
from mpl_toolkits.mplot3d import Axes3D
from scipy.ndimage import gaussian_filter

# Image dimensions
image_size = 256

# Circular tumor properties
tumor_radius = int(image_size / 6)
tumor_center = (int(image_size / 2), int(image_size / 2), int(image_size / 2))

# Define cell clusters properties within the tumor. Use uptake range to determine heterogeneity
num_circles = 3000
circle_radius_range = (2, 5)
circle_uptake_range = (50, 255)  # Update the range to 0-255 levels of gray

# Define point spread function (PSF). Vary as need be
psf_sigma = 3.0

# Create a black background volume
volume = np.zeros((image_size, image_size, image_size), dtype=np.float32)

# Generate the tumor
y, x, z = np.ogrid[:image_size, :image_size, :image_size]
tumor_mask = (x - tumor_center[0]) ** 2 + (y - tumor_center[1]) ** 2 + (z - tumor_center[2]) ** 2 <= tumor_radius ** 2

# Generate the vell clusters
for _ in range(num_circles):
    # Generate random circle properties
    circle_radius = np.random.uniform(*circle_radius_range)
    circle_uptake = int(np.random.uniform(*circle_uptake_range))  # Convert uptake value to integer

    # Generate random cell cluster positions within the tumor
    while True:
        circle_x = np.random.randint(tumor_center[0] - tumor_radius + circle_radius,
                                     tumor_center[0] + tumor_radius - circle_radius + 1)
        circle_y = np.random.randint(tumor_center[1] - tumor_radius + circle_radius,
                                     tumor_center[1] + tumor_radius - circle_radius + 1)
        circle_z = np.random.randint(tumor_center[2] - tumor_radius + circle_radius,
                                     tumor_center[2] + tumor_radius - circle_radius + 1)

        # Check if the circle is inside the tumor mask
        if tumor_mask[circle_x, circle_y, circle_z]:
            break

    # Generate the circle mask
    circle_mask = (x - circle_x) ** 2 + (y - circle_y) ** 2 + (z - circle_z) ** 2 <= circle_radius ** 2

    # Apply the circle mask with the corresponding uptake value to the volume
    volume[circle_mask] = circle_uptake

# Apply Gaussian blurring with the PSF
volume_blurred = gaussian_filter(volume, sigma=psf_sigma)

# Define DICOM pixel spacing
voxel_size = 1.5  # Voxel size in millimeters
pixel_spacing = [voxel_size, voxel_size, voxel_size]

# Define DICOM Object position
patient_position = "HFS"  # Head First Supine

# Create a FileDataset for PET image
ds_pet = FileDataset("", {}, file_meta=Dataset(), preamble=b"\0" * 128)

# Set File Meta Information elements
ds_pet.file_meta.MediaStorageSOPClassUID = pydicom.uid.generate_uid()
ds_pet.file_meta.TransferSyntaxUID = pydicom.uid.ExplicitVRLittleEndian

# Set common DICOM attributes for PET image
ds_pet.SOPInstanceUID = pydicom.uid.generate_uid()
ds_pet.SOPClassUID = pydicom.uid.PositronEmissionTomographyImageStorage
ds_pet.StudyInstanceUID = "1.2.3.4"
ds_pet.SeriesInstanceUID = pydicom.uid.generate_uid()
ds_pet.SeriesNumber = 1
ds_pet.Modality = "PT"
ds_pet.PatientName = "Simulation"
ds_pet.PatientID = "123456"
ds_pet.PixelSpacing = pixel_spacing
ds_pet.PatientPosition = patient_position
ds_pet.ImagePositionPatient = [0, 0, 0]
ds_pet.SliceThickness = voxel_size
ds_pet.Rows, ds_pet.Columns, ds_pet.NumberOfFrames = volume_blurred.shape
ds_pet.PixelData = volume_blurred.astype(np.uint16).tobytes()
ds_pet.SamplesPerPixel = 1
ds_pet.PhotometricInterpretation = "MONOCHROME2"
ds_pet.BitsAllocated = 16
ds_pet.BitsStored = 16
ds_pet.HighBit = 15
ds_pet.PixelRepresentation = 0  # Unsigned integer

# Create a unique filename for the DICOM file
filename_pet = "XYZ.dcm"

# Save the PET DICOM file
ds_pet.save_as(filename_pet, write_like_original=False)

print(f"Saved PET image as DICOM file: {filename_pet}")

# Extract and visualize coronal, sagittal, and axial slices
coronal_slice = volume_blurred[:, int(image_size / 2), :]
sagittal_slice = volume_blurred[int(image_size / 2), :, :]
axial_slice = volume_blurred[:, :, int(image_size / 2)]

plt.figure(figsize=(10, 10))

#plt.subplot(1, 3, 1)
#plt.imshow(coronal_slice, cmap="gray")
#plt.title("Coronal Slice")
#plt.axis("off")

#plt.subplot(1, 3, 2)
#plt.imshow(sagittal_slice, cmap="gray")
#plt.title("Sagittal Slice")
#plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(axial_slice, cmap="gray")
#plt.title("L1")
plt.axis("off")

#plt.tight_layout()
#plt.show()

# Save the figure with high DPI (dots per inch) for better quality
plt.tight_layout()
plt.savefig('XYZ', dpi=300)
plt.show()