import numpy as np
import tensorflow as tf


def nearest_upsampling(data, scale):
    """ Nearest neighbor upsampling implementation.
  Args:
    data: A float32 tensor of size [batch, height_in, width_in, channels].
    scale: An integer multiple to scale resolution of input data.
  Returns:
    data_up: A float32 tensor of size
      [batch, height_in*scale, width_in*scale, channels].
  """
    with tf.name_scope('nearest_upsampling'):
        bs, h, w, c = data.get_shape().as_list()
        bs = -1 if bs is None else bs
        # Use reshape to quickly upsample the input.  The nearest pixel is selected
        # implicitly via broadcasting.
        data = tf.reshape(data, [bs, h, 1, w, 1, c]) * tf.ones(
            [1, 1, scale, 1, scale, 1], dtype=data.dtype)
        return tf.reshape(data, [bs, h * scale, w * scale, c])


def transpose_shape(shape, target_format, spatial_axes):
    """Converts a tuple or a list to the correct `data_format`.
    It does so by switching the positions of its elements.
    # Arguments
        shape: Tuple or list, often representing shape,
            corresponding to `'channels_last'`.
        target_format: A string, either `'channels_first'` or `'channels_last'`.
        spatial_axes: A tuple of integers.
            Correspond to the indexes of the spatial axes.
            For example, if you pass a shape
            representing (batch_size, timesteps, rows, cols, channels),
            then `spatial_axes=(2, 3)`.
    # Returns
        A tuple or list, with the elements permuted according
        to `target_format`.
    # Example
    # Raises
        ValueError: if `value` or the global `data_format` invalid.
    """
    if target_format == 'channels_first':
        new_values = shape[:spatial_axes[0]]
        new_values += (shape[-1],)
        new_values += tuple(shape[x] for x in spatial_axes)

        if isinstance(shape, list):
            return list(new_values)
        return new_values
    elif target_format == 'channels_last':
        return shape
    else:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(target_format))


def permute_dimensions(x, pattern):
    """Permutes axes in a tensor.
    # Arguments
        x: Tensor or variable.
        pattern: A tuple of
            dimension indices, e.g. `(0, 2, 1)`.
    # Returns
        A tensor.
    """
    return tf.transpose(x, perm=pattern)


def int_shape(x):
    """Returns the shape of tensor or variable as a tuple of int or None entries.
    # Arguments
        x: Tensor or variable.
    # Returns
        A tuple of integers (or None entries).
    """
    if hasattr(x, '_keras_shape'):
        return x._keras_shape
    try:
        return tuple(x.get_shape().as_list())
    except ValueError:
        return None


def resize_images(x,
                  height_factor,
                  width_factor,
                  data_format,
                  interpolation='nearest'):
    """Resizes the images contained in a 4D tensor.
    # Arguments
        x: Tensor or variable to resize.
        height_factor: Positive integer.
        width_factor: Positive integer.
        data_format: string, `"channels_last"` or `"channels_first"`.
        interpolation: A string, one of `nearest` or `bilinear`.
    # Returns
        A tensor.
    # Raises
        ValueError: if `data_format` is neither `"channels_last"` or `"channels_first"`.
    """
    if data_format == 'channels_first':
        rows, cols = 2, 3
    else:
        rows, cols = 1, 2

    original_shape = int_shape(x)
    new_shape = tf.shape(x)[rows:cols + 1]
    new_shape *= tf.constant(np.array([height_factor, width_factor], dtype='int32'))

    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 2, 3, 1])
    if interpolation == 'nearest':
        x = tf.image.resize(x, new_shape, method=tf.image.ResizeMethod.BILINEAR)
    elif interpolation == 'bilinear':
        x = tf.image.resize(x, new_shape, method=tf.image.ResizeMethod.BILINEAR)
    else:
        raise ValueError('interpolation should be one '
                         'of "nearest" or "bilinear".')
    if data_format == 'channels_first':
        x = permute_dimensions(x, [0, 3, 1, 2])

    if original_shape[rows] is None:
        new_height = None
    else:
        new_height = original_shape[rows] * height_factor

    if original_shape[cols] is None:
        new_width = None
    else:
        new_width = original_shape[cols] * width_factor

    output_shape = (None, new_height, new_width, None)
    x.set_shape(transpose_shape(output_shape, data_format, spatial_axes=(1, 2)))
    return x
