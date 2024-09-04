"""Definition of the datasets and associated functions."""
import os
import math
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

__all__ = [
  "get_slice_points",
  "reconstruct_patched",
  "collate_fn",
  "MicrographDataset",
  "MicrographDatasetSingle",
  "MicrographDatasetEvery"
]

def create_weight_map(crop_size, bandwidth=None):
    """Create a Gaussian-like weight map which emphasizes the center of the patch.
    
    Args:
        crop_size (int): The size of the crop (assumed to be square for simplicity).
        bandwidth (int, optional): Defines the transition area from edge to center.
                                   Smaller values make the transition sharper.
                                   If None, it defaults to crop_size // 4.
    """
    if bandwidth is None:
        bandwidth = crop_size // 16  # Default bandwidth

    ramp = torch.linspace(0, 1, bandwidth)
    ramp = torch.cat([ramp, torch.ones(crop_size - 2 * bandwidth), ramp.flip(0)])
    weight_map = ramp[:, None] * ramp[None, :]
    return weight_map


def get_slice_points(image_size, crop_size, overlap):
    """ Calculate the slice points for cropping the image into overlapping patches. """
    step = crop_size - overlap
    num_points = (image_size - crop_size) // step + 1
    last_point = image_size - crop_size
    points = torch.arange(0, step * num_points, step)
    if points[-1] != last_point:
        points = torch.cat([points, torch.tensor([last_point])])
    return points


def reconstruct_patched(images, structured_grid, bandwidth=None):
    if bandwidth is None:
        bandwidth = images.shape[2] // 16  # Adjust this based on your needs

    weight_map = create_weight_map(images.shape[2], bandwidth).to(images.device)
    max_height = structured_grid[0, -1] + images.shape[2]
    max_width = structured_grid[1, -1] + images.shape[3]
    reconstructed_image = torch.zeros((images.shape[1], max_height, max_width), device=images.device)
    weights = torch.zeros_like(reconstructed_image)

    # Process in batches
    batch_size = 32  # Adjust this depending on your GPU capacity
    num_batches = (images.shape[0] + batch_size - 1) // batch_size  # Compute number of batches

    for batch_index in range(num_batches):
        batch_start = batch_index * batch_size
        batch_end = min(batch_start + batch_size, images.shape[0])
        batch_images = images[batch_start:batch_end]
        batch_structured_grid = structured_grid[:, batch_start:batch_end]

        for idx, (start_i, start_j) in enumerate(zip(batch_structured_grid[0].flatten(), batch_structured_grid[1].flatten())):
            end_i = start_i + images.shape[2]
            end_j = start_j + images.shape[3]
            reconstructed_image[:, start_i:end_i, start_j:end_j] += batch_images[idx] * weight_map
            weights[:, start_i:end_i, start_j:end_j] += weight_map

    reconstructed_image /= weights.clamp(min=1)
    return reconstructed_image


def collate_fn(batch):
    images, masks = [], []
    for b in batch:
        for image, mask in b:
            images.append(image)
            masks.append(mask)
    return torch.stack(images), torch.stack(masks)

class MicrographDataset(Dataset):
  """
  Dataset for cryo-EM dataset that returns multiple patches per image.
  """
  def __init__(self, image_dir, label_dir, filenames=None, crop_size=(512, 512), num_patches=1, img_ext='.npy', crop=None):
      self.image_dir = image_dir
      self.label_dir = label_dir
      self.num_patches = num_patches
      self.crop_size = crop_size
      if filenames is not None:
          self.filenames = filenames
      else:
          self.filenames = sorted(os.listdir(image_dir))
      basenames = [os.path.splitext(filename)[0] for filename in self.filenames]
      self.images = [os.path.join(image_dir, basename + img_ext) for basename in basenames]
      self.labels = [os.path.join(label_dir, basename + '.png') for basename in basenames]
      if crop is None:
          self.crop = transforms.CenterCrop(3840)  # Adjust based on your specific needs
      else:
          self.crop = crop

  def __len__(self):
      return len(self.images)

  def __getitem__(self, idx):
      mask = TF.to_tensor(Image.open(self.labels[idx]).convert("L"))
      image = torch.from_numpy(np.load(self.images[idx]).reshape((-1, mask.shape[1], mask.shape[2])))  # Assume images are 4096x4096

      patches = []
      for _ in range(self.num_patches):
          image_cropped, mask_cropped = self.transform(image, mask)
          patches.append((image_cropped, mask_cropped.long()))
      
      return patches

  def transform(self, image, mask):
      if self.crop:
          image = self.crop(image)
          mask = self.crop(mask)
      
      i, j, h, w = transforms.RandomCrop.get_params(image, output_size=self.crop_size)
      image = TF.crop(image, i, j, h, w)
      mask = TF.crop(mask, i, j, h, w)
      #mask = torch.concat([1 - mask, mask], dim=0)  # For two-class problem: background and foreground
      
      return image, mask

class MicrographDatasetSingle(Dataset):
  """
  Dataset for cryo-EM dataset.
  The micrographs and ground truths will be random crop to `crop_size`.
  """
  def __init__(self, image_dir, label_dir, filenames=None, crop_size=(512, 512), img_ext='.npy', crop=3840):
    self.image_dir = image_dir
    self.label_dir = label_dir
    if filenames is not None:
      self.filenames = filenames
    else:
      self.filenames = sorted(os.listdir(image_dir))
    basenames = [os.path.splitext(filename)[0] for filename in filenames]
    self.images = [os.path.join(image_dir, basename+img_ext) for basename in basenames]
    self.labels = [os.path.join(label_dir, basename+'.png') for basename in basenames]
    if crop: # To be formalized.
      self.crop = transforms.CenterCrop(3840) # = 4096-256, uses because of the property of EMPIAR-10017
    else:
      self.crop = None
    self.crop_size = crop_size

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    mask = TF.to_tensor(Image.open(self.labels[idx]).convert("L"))
    image = torch.from_numpy(np.load(self.images[idx]).reshape((-1,mask.shape[1], mask.shape[2]))) # (4096, 4096) is the image size of micrographs EMPIAR-10017
    #mask = mask.long()
    return self.transform(image, mask)

  def transform(self, image, mask):
    if self.crop:
        image = self.crop(image)
        mask = self.crop(mask)

    i, j, h, w = transforms.RandomCrop.get_params(
      image, output_size=self.crop_size)
    image = TF.crop(image, i, j, h, w)
    mask = TF.crop(mask, i, j, h, w)
    #mask = torch.concat([1-mask, mask], dim=0) # Remove this line if background is not consider.
    return image, mask.long()

class MicrographDatasetEvery(MicrographDatasetSingle):
  """
  Dataset for cryo-EM dataset.
  The micrographs and ground truths will be divided into grid.
  """
  def __init__(self, *arg, **kwarg):
    super().__init__(*arg, **kwarg)

  def transform(self, image, mask):
    if self.crop:
        image = self.crop(image)
        mask = self.crop(mask) #CenterCrop

    image_dims = (image.size(-2), image.size(-1))
    crop_dims = (self.crop_size[-2], self.crop_size[-1])
    overlap_size = 64
    # Cache grid calculations to avoid redundancy
    #if (image_dims, crop_dims) not in self.slice_points_cache:
    grid_i = get_slice_points(image.size(-2), self.crop_size[-2], overlap_size)
    grid_j = get_slice_points(image.size(-1), self.crop_size[-1], overlap_size)
    grid = torch.cartesian_prod(grid_i, grid_j)
    #self.slice_points_cache[(image_dims, crop_dims)] = grid

    # Pre-allocate tensors for images and masks
    num_patches = grid.size(0)
    images = torch.zeros((num_patches, 1, *self.crop_size), device=image.device, dtype=image.dtype)
    masks = torch.zeros((num_patches, 1, *self.crop_size), device=mask.device, dtype=mask.dtype)

    for idx, (i, j) in enumerate(grid):
        images[idx] = TF.crop(image, i.item(), j.item(), self.crop_size[-2], self.crop_size[-1])
        masks[idx] = TF.crop(mask, i.item(), j.item(), self.crop_size[-2], self.crop_size[-1])

    structured_grid = torch.stack([grid[:, 0], grid[:, 1]], dim=0)

    #masks = torch.concat([1-masks, masks], dim=1) # Remove this line if background is not consider.

    return images, masks.long(), structured_grid, mask.long()