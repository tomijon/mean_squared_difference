from os import environ
from time import time as current_time

from numpy import dtype, array, zeros
from pyopencl import *
from PIL import Image

# Easy access mem flags.
WRITE_ONLY = mem_flags.WRITE_ONLY
READ_ONLY = mem_flags.READ_ONLY
COPY_HOST_PTR = mem_flags.COPY_HOST_PTR

# Set default mode for pyopencl to use.
environ["PYOPENCL_CTX"] = "0"

# Custom data type to allow for guaranteed int3 values instead of using int4
vector3 = np.dtype([('x', np.int32), ('y', np.int32), ('z', np.int32)])


def msd(image1: Image, image2: Image):
    """Calculate the mean squared difference of the two images.

    Parameters:
        image1 - A Pil.Image object
        image2 - A Pil.Image object

    Returns:
        A float representing the mean squared difference of the two
        provided images.
    """
    # Convert Images to numpy array.
    image1_np = np.array(image1.getdata(), dtype=vector3)
    image2_np = np.array(image2.getdata(), dtype=vector3)

    # Squared difference arrays.
    sd = np.zeros(image1_np.shape, dtype=np.int32)
    sd_gpu = Buffer(context, WRITE_ONLY, sd.nbytes)

    # Transfer data in image numpy arrays to buffer in GPU memory.
    image1_gpu = Buffer(context, READ_ONLY | COPY_HOST_PTR, hostbuf=image1_np)
    image2_gpu = Buffer(context, READ_ONLY | COPY_HOST_PTR, hostbuf=image2_np)

    # Gather the squared difference and perform the average calculation.
    squared_difference(context_queue, image1_np.shape, None,
                       image1_gpu, image2_gpu, sd_gpu)
    enqueue_copy(context_queue, sd, sd_gpu)
    return sd.mean()


# Create opencl context and queue.
context = create_some_context()
context_queue = CommandQueue(context)

# Function for calculating the squared difference between every pixel.
squared_difference = Program(context, """
// Vector3 struct the same as the one defined using numpy.
typedef struct vector3 {
    int x;
    int y;
    int z;
} Vector3;

__kernel void squared_difference(
    __global const Vector3 *image1_pixels,
    __global const Vector3 *image2_pixels,
    __global int *squared_difference)
{
    int gid = get_global_id(0);

    // Calculate pixel intensity.
    int image1_intensity = (int) ((0.299 * image1_pixels[gid].x)
                                  + (0.587 * image1_pixels[gid].y)
                                  + (0.114 * image1_pixels[gid].z));
    int image2_intensity = (int) ((0.299 * image2_pixels[gid].x)
                                  + (0.587 * image2_pixels[gid].y)
                                  + (0.114 * image2_pixels[gid].z));

    // Calculate squared difference.
    int sd = ((image1_intensity - image2_intensity)
              * (image1_intensity - image2_intensity));
  
    squared_difference[gid] = sd;
}
""").build()
squared_difference = squared_difference.squared_difference

# Open both images for comparisment.
image1 = Image.open("image0.png").convert("RGB")
image2 = Image.open("image1.png").convert("RGB")

# Time the calculation.
start = current_time()
mean = msd(image1, image2)
elapsed = current_time() - start

# Output.
print(f"MSD = {round(mean, 3)}")
print(f"Time Taken: {round(elapsed, 3)}s")
