from copy import deepcopy
import numpy as np
import numpy.ma as ma
from scipy.ndimage import label
from skimage.segmentation import flood_fill

def corner_crop(x, src_start, patch_size, default_fill = 0.0):
    def _clamp(v, a, b):
        s=deepcopy(v)
        if s<a: s=a
        if s>b: s=b
        return s
    def _clamp3(v,a,b):
        s=[0,0,0]
        s[0] = _clamp(v[0],a[0],b[0])
        s[1] = _clamp(v[1],a[1],b[1])
        s[2] = _clamp(v[2],a[2],b[2])
        return s
    def _add3(a,b):
        return [a[0]+b[0],a[1]+b[1],a[2]+b[2]]
    def _sub3(a,b):
        return [a[0]-b[0],a[1]-b[1],a[2]-b[2]]
    src_end = [
        src_start[0] + patch_size[0],
        src_start[1] + patch_size[1],
        src_start[2] + patch_size[2]
    ]
    src_actual_start = _clamp3(src_start, (0,0,0), x.shape)
    src_actual_end   = _clamp3(src_end,   (0,0,0), x.shape)
    src_actual_size  = _sub3(src_actual_end, src_actual_start)
    dst_start = _sub3(src_actual_start, src_start)
    dst_end = _add3(dst_start, src_actual_size)
    y = np.zeros(patch_size)
    y.fill(default_fill)
    y[dst_start[0]:dst_end[0], dst_start[1]:dst_end[1], dst_start[2]:dst_end[2]] = \
        x[src_actual_start[0]:src_actual_end[0], src_actual_start[1]:src_actual_end[1], src_actual_start[2]:src_actual_end[2]]
    return y

def center_crop(x, center_pos, patch_size, default_fill = 0.0):
    src_start = [ center_pos[0] - patch_size[0]//2, center_pos[1] - patch_size[1]//2, center_pos[2] - patch_size[2]//2 ]
    return corner_crop(x, src_start, patch_size, default_fill=default_fill)

def masked_mean(data, mask):
    mask = (mask>0.5).astype('int')
    masked_data = ma.masked_array(data,mask=1-mask)
    return masked_data.mean()

def masked_std(data, mask):
    mask = (mask>0.5).astype('int')
    masked_data = ma.masked_array(data,mask=1-mask)
    return masked_data.std()

def z_score(data, mask=None):
    '''
    perform z-score normalization for image data.
    '''
    data_mean = np.mean(data) if mask is None else masked_mean(data, mask)
    data_std = np.std(data) if mask is None else masked_std(data, mask)
    data_std = np.max( [ data_std, 0.00001 ] ) # avoid division by zero
    return (data - data_mean) / data_std

def barycentric_coordinate(weights: np.ndarray):
    '''
    calculate barycentric coordinate for a given 3D image
    '''
    assert len(weights.shape) == 3, 'must be a 3D image.'
    x = np.sum(np.sum(weights, (1,2)) * np.arange(weights.shape[0])) / np.sum(weights)
    y = np.sum(np.sum(weights, (0,2)) * np.arange(weights.shape[1])) / np.sum(weights)
    z = np.sum(np.sum(weights, (0,1)) * np.arange(weights.shape[2])) / np.sum(weights)
    return x,y,z

def connected_components(mask, return_labeled=True):
    '''
    Description
    -----------
    Get number of connected components and their volumes.
    0 is considered as background and is not counted in 
    connected components. If "return_volumes" is True, a
    list will be returned containing volumes of each component,
    otherwise a total number of connected component (int) 
    is returned.

    Usage
    -----------
    >>> num_parts, labeled_array = connected_comps(mask)
    >>> num_parts = connected_comps(mask, return_labeled = False)
    '''
    mask = (mask>0.5).astype('int')
    labeled_array, num_parts = label(mask)
    if return_labeled:
        return num_parts, labeled_array
    else:
        return num_parts

def max_volume_filter(mask, return_type = 'float32'):
    num_parts, labeled_array = connected_components(mask)
    max_vol_id, max_volume = 0, 0
    for part_id in range(1, num_parts+1):
        volume = np.sum((labeled_array == part_id).astype('int32'))
        if volume > max_volume:
            max_volume = volume
            max_vol_id = part_id
    return (labeled_array == part_id).astype(return_type)

def remove_sparks(mask, min_volume=3, verbose=False):
    '''
    remove sparks for a given (binarized) image.
    any component smaller than min_volume will be discarded.
    '''
    mask = (mask>0.5).astype('int')
    if verbose:
        print('calculating cc...')
    labeled_array, num_features = label(mask)
    if verbose:
        print('%d cc detected.' % num_features)
    filtered_mask = np.zeros_like(mask) 
    for i in range(1, num_features+1):
        v = (labeled_array==i).sum()
        if v>=min_volume:
            filtered_mask[labeled_array==i] = 1
    if verbose:
        _, n = label(filtered_mask)
        print('cc after filtering: %d.' % n)
    return filtered_mask

def find_holes(mask:np.ndarray):
    '''
    Find holes in a 3D image.
    '''
    assert len(mask.shape) == 3, 'mask must be a 3D image.'

    mask = (mask > 0.5).astype('int32')
    x = np.zeros( [ mask.shape[0]+2, mask.shape[1]+2, mask.shape[2]+2 ] ).astype('int32')
    y = np.zeros_like(x)
    # pad zeros around mask, and treat background of the original image as foreground
    x[1:1+mask.shape[0], 1:1+mask.shape[1], 1:1+mask.shape[2]] = mask+1
    y[1:1+mask.shape[0], 1:1+mask.shape[1], 1:1+mask.shape[2]] = mask
    holes = np.zeros_like(x)
    num_components, labeled = connected_components(x)
    for i in range(1, num_components+1):
        selected = (labeled == i).astype('int32')
        # for each labeled component, we check if it is the background of the original image
        if np.sum(y * selected) > 0:
            continue # no, it is the foreground
        # ok, we found a background region, using flood fill to detect if it is closed
        position = np.argwhere(selected>0)[0]
        z = flood_fill(x, position, -1)
        if z[0,0,0] == -1:
            # this background region is not a hole
            continue
        # ok, now we find a hole
        holes += selected
    return holes

def make_onehot_from_label(label: np.ndarray):
    label = label.astype('int32')
    max_label_id = np.max(label)
    num_channels = max_label_id + 1
    y = np.zeros([num_channels, *label.shape]).astype('float32')
    for channel_id in range(num_channels):
        y[channel_id] = (label == channel_id)
    return y
