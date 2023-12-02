import math
import random
import numpy as np
from numba import jit, njit, cuda, prange
from PIL import Image


# Define a function to compute the intensity of light on a surface using a basic diffuse lighting model
@njit
def calculate_light_intensity(normal_vector, light_parameters):
    # Normalize the vector
    normal_vector = normal_vector / abs(normal_vector)

    # Calculate simple diffuse light component
    diffuse_light = (normal_vector.real * math.cos(light_parameters[0]) +
                     normal_vector.imag * math.sin(light_parameters[0]))
    
    # Adjust the diffuse light component based on light intensity
    adjusted_diffuse_light = diffuse_light * light_parameters[1]

    return max(0, min(1, adjusted_diffuse_light))


# Define a function to create a color table using sine waves for visualizing the Mandelbrot set.
@njit
def colortable(rgb=(.85, .0, .15), column_num=2 ** 12):
    def colormap(x, rgb):
        # Calculate color values based on sine functions.
        y = np.column_stack(((x + rgb[0]) * 2 * math.pi,
                             (x + rgb[1]) * 2 * math.pi,
                             (x + rgb[2]) * 2 * math.pi))
        return 0.5 + 0.5 * np.sin(y)

    # Return a colormap array for a linearly spaced range of values.
    return colormap(np.linspace(0, 1, column_num), rgb)


# Define a function to calculate the smooth iteration count for each point in the Mandelbrot set.
@njit(parallel=True)
def smooth_iter(c, max_iter):
    esc_radius_squared = 10 ** 10  # Escape radius squared, used to determine if a point has escaped.
    z = 0j  # Initialize z at the origin of the complex plane.
    dz = 1+0j # Initialize derivative of z.

    for n in prange(max_iter):
        z = z * z + c  # Mandelbrot's iteration.
        dz = dz * 2 * z + 1 # Update the derivative of z.

        # Check if the magnitude of z squared is greater than the escape radius squared.
        if z.real * z.real + z.imag * z.imag > esc_radius_squared:
            mod_z = abs(z)  # Calculate the magnitude of z.
            # Calculate logarithmic smoothing to get a smoother color transition.
            log_ratio = 2 * math.log(mod_z) / math.log(esc_radius_squared)
            smooth_i = 1 - math.log(log_ratio) / math.log(2)

            # Return the smoothed iteration count and dz.
            return n + smooth_i, z/dz

    # Return default value if the point does not escape.
    return 0


# Define a function to color a pixel based on the iteration count.
@jit
def color_pixel(matrix, n_iter, colortable, n_cycle, normal_vector, lightning):
    n_col = colortable.shape[0] - 1  # Number of colors in the color table.
    n_iter = math.sqrt(n_iter) % n_cycle / n_cycle  # Normalize iteration count.
    col_i = round(n_iter * n_col)  # Index for the color table.
    
    # Calculate the brightness based on the normal vector and lighting parameters.
    brightness = calculate_light_intensity(normal_vector, lightning)

    # Assign a color from the color table to the pixel and ensure it is within [0, 1] range.
    for i in range(3):
        matrix[i] = colortable[col_i, i] * brightness # Apply brightness to the selected color
        matrix[i] = max(0, min(1, matrix[i]))


# Define a function to compute the Mandelbrot set and color it.
@jit(parallel=True)
def compute_set(real_part, imaginary_part, max_iter, colortable, n_cycle, lightning):
    x_pixels = len(real_part)  # Width of the image in pixels.
    y_pixels = len(imaginary_part)  # Height of the image in pixels.
    mat = np.zeros((y_pixels, x_pixels, 3))  # Initialize the image matrix.

    # Iterate over each pixel in the image.
    for x in prange(x_pixels):
        for y in range(y_pixels):
            c = complex(real_part[x], imaginary_part[y])  # Represent the pixel as a complex number.
            # Calculate the iteration count and normal vector for this point.
            n_iter, normal_vector = smooth_iter(c, max_iter)
            # Color the pixel if the iteration count is greater than 0.
            if n_iter > 0:
                color_pixel(mat[y, x], n_iter, colortable, n_cycle, normal_vector, lightning)
    return mat  # Return the colored image matrix.


# Define a function to compute the Mandelbrot set using GPU acceleration.
@cuda.jit
def compute_set_gpu(matrix, x_min, x_max, y_min, y_max, max_iter, colortable, n_cycle, lightning):
    index = cuda.grid(1)  # Get the index of the current thread in the grid.
    x, y = index % matrix.shape[1], index // matrix.shape[1]  # Calculate x and y coordinates.

    # Process each pixel if it is within the image bounds.
    if (y < matrix.shape[0]) and (x < matrix.shape[1]):
        # Calculate the real and imaginary parts of the complex number for this pixel.
        c_real = x_min + x / (matrix.shape[1] - 1) * (x_max - x_min)
        cim = y_min + y / (matrix.shape[0] - 1) * (y_max - y_min)
        c = complex(c_real, cim)
        # Calculate the iteration count and normal vector for this point.
        n_iter, normal_vector = smooth_iter(c, max_iter)
        # Color the pixel if the iteration count is greater than 0.
        if n_iter > 0:
            color_pixel(matrix[y, x], n_iter, colortable, n_cycle, normal_vector, lightning)


# Define a class for generating and visualizing Mandelbrot sets.
class Mandelbrot:
    def __init__(self, x_pixels=1280, max_iterations=500, coord=None, gpu=True, n_cycle=32,
                 rgb=(.0, .15, .25), oversampling=3, lightning = (45., 45.)):

        # If no coordinates are provided, generate random coordinates for the Mandelbrot set.
        if coord is None:
            x_min, x_max = -2.0, 0.6  # Define x-axis range.
            y_min, y_max = -1.3, 1.3  # Define y-axis range.
            x_center = random.uniform(x_min, x_max)
            y_center = random.uniform(y_min, y_max)
            min_width, max_width = 0.1, 1.0
            range_width = random.uniform(min_width, max_width)
            self.coord = [x_center - range_width, x_center + range_width,
                          y_center - range_width, y_center + range_width]
        else:
            self.coord = coord

        # Initialize various parameters for the Mandelbrot set visualization.
        self.x_pixels = x_pixels
        self.max_iterations = max_iterations
        self.gpu = gpu
        self.n_cycle = n_cycle
        self.os = oversampling
        self.rgb = rgb
        self.y_pixels = round(self.x_pixels / (self.coord[1] - self.coord[0]) * (self.coord[3] - self.coord[2]))
        self.colortable = colortable(self.rgb)
        self.lightning = np.array(lightning)
        # Convert from degrees to radians.
        self.lightning[0] = 2 * math.pi * self.lightning[0] / 360
        self.lightning[1] = math.pi / 2 * self.lightning[1] / 90
        self.update_set()

    def update_set(self):
        # Normalize the number of color cycles and calculate the diagonal length of the view area.
        # This is used in scaling and smoothing the image.
        n_cycle = math.sqrt(self.n_cycle)

        # Calculate the pixel dimensions of the image, considering oversampling.
        xp = self.x_pixels * self.os  # Oversampled width
        yp = self.y_pixels * self.os  # Oversampled height

        # Check if GPU acceleration is enabled.
        if self.gpu:
            # Initialize the image array to zeros.
            self.set = np.zeros((yp, xp, 3))

            # Calculate the total number of pixels and the grid configuration for CUDA.
            n_pixels = xp * yp
            n_thread = 32  # Number of threads per block.
            n_block = math.ceil(n_pixels / n_thread)  # Number of blocks.

            # Call the GPU-accelerated function to compute the Mandelbrot set.
            compute_set_gpu[n_block, n_thread](self.set, *self.coord, self.max_iterations, self.colortable, n_cycle, self.lightning)
        else:
            # If GPU is not used, compute the set using the CPU.
            # Generate linearly spaced values for the real and imaginary components.
            real = np.linspace(self.coord[0], self.coord[1], xp)
            im = np.linspace(self.coord[2], self.coord[3], yp)

            # Compute the Mandelbrot set for each pixel.
            self.set = compute_set(real, im, self.max_iterations,
                                   self.colortable, n_cycle, self.lightning)

        # Convert the set to an 8-bit format suitable for image generation.
        self.set = (255 * self.set).astype(np.uint8)

        # If oversampling is used, resize the image to its original dimensions by averaging.
        if self.os > 1:
            self.set = (self.set
                        .reshape((self.y_pixels, self.os, self.x_pixels, self.os, 3))
                        .mean(3).mean(1).astype(np.uint8))

    def draw(self, filename=None):
        # Convert the computed set to an image and either save or display it.
        img = Image.fromarray(self.set[::-1, :, :], 'RGB')
        if filename is not None:
            img.save(filename)
        else:
            img.show()


# Main execution block for generating a Mandelbrot set image.
if __name__ == "__main__":
    mand = Mandelbrot(max_iterations=5000, rgb=[.45, .0, .13])
    mand.draw()  # Create and display the Mandelbrot set image.
