import math
import numpy as np
from numba import jit, njit, cuda, prange
import imageio

# Function to create a color palette for the fractal.
@njit
def create_color_palette(angle_offsets=(.85, .0, .15), total_colors=2**12):
    # Palette generator function, creating a color mapping based on sine wave modulation.
    def palette_generator(value, angle_offsets):
        angles = np.column_stack(((value + angle_offsets[0]) * 2 * math.pi,
                                  (value + angle_offsets[1]) * 2 * math.pi,
                                  (value + angle_offsets[2]) * 2 * math.pi))
        return 0.5 + 0.5 * np.sin(angles)
    # Generates a linearly spaced array of colors.
    return palette_generator(np.linspace(0, 1, total_colors), angle_offsets)

# Function to calculate iterations for each point in the Mandelbrot set.
@njit(parallel=True)
def calculate_iterations(complex_number, iteration_limit):
    # Set escape limit for determining divergence.
    escape_limit = 10**10
    z_value = 0j
    derivative_z = 1 + 0j
   
    # Iterate and check if the point escapes the Mandelbrot set.
    for iteration in prange(iteration_limit):
        derivative_z = derivative_z * 2 * z_value + 1
        z_value = z_value * z_value + complex_number
       
        # Check for escape and calculate smooth coloring.
        if z_value.real * z_value.real + z_value.imag * z_value.imag > escape_limit:
            modulus_z = abs(z_value)
            logarithmic_ratio = 2 * math.log(modulus_z) / math.log(escape_limit)
            smooth_iteration = 1 - math.log(logarithmic_ratio) / math.log(2)

            return iteration + smooth_iteration
           
    return 0
           
# Function to apply color to a pixel based on iteration count.
@jit
def apply_color(pixel_matrix, iteration_count, color_palette, color_cycle):
    number_of_colors = color_palette.shape[0] - 1
    normalized_iteration = math.sqrt(iteration_count) % color_cycle / color_cycle
    color_index = round(normalized_iteration * number_of_colors)

    # Apply color from the palette to the pixel.
    for i in range(3):
        pixel_matrix[i] = color_palette[color_index, i]
        pixel_matrix[i] = max(0, min(1, pixel_matrix[i]))
        
# Function to generate the Mandelbrot set fractal.
@jit(parallel=True)
def generate_fractal(real_axis, imaginary_axis, max_iterations, color_palette, color_cycle):
    width = len(real_axis)
    height = len(imaginary_axis)
    fractal_matrix = np.zeros((height, width, 3))

    # Compute the fractal for each pixel.
    for x in prange(width):
        for y in range(height):
            complex_value = complex(real_axis[x], imaginary_axis[y])
            iteration_count = calculate_iterations(complex_value, max_iterations)
            if iteration_count > 0:
                apply_color(fractal_matrix[y, x,], iteration_count, color_palette, color_cycle)
    return fractal_matrix

# GPU-accelerated function to generate the fractal.
@cuda.jit
def generate_fractal_gpu(fractal_matrix, min_real, max_real, min_imag, max_imag, max_iterations, color_palette, color_cycle):
    thread_index = cuda.grid(1)
    x, y = thread_index % fractal_matrix.shape[1], thread_index // fractal_matrix.shape[1]

    # Compute the fractal for each pixel using GPU parallelization.
    if (y < fractal_matrix.shape[0]) and (x < fractal_matrix.shape[1]):
        real_value = min_real + x / (fractal_matrix.shape[1] - 1) * (max_real - min_real)
        imag_value = min_imag + y / (fractal_matrix.shape[0] - 1) * (max_imag - min_imag)
        complex_value = complex(real_value, imag_value)
        iteration_count = calculate_iterations(complex_value, max_iterations)
        if iteration_count > 0:
            apply_color(fractal_matrix[y, x,], iteration_count, color_palette, color_cycle)

# Class to handle the creation and manipulation of the Mandelbrot set.
class FractalExplorer():
    def __init__(self, width=1280, iteration_limit=500, coordinates=(-2.6, 1.845, -1.25, 1.25), use_gpu=True, cycle_length=32, angle_offsets=(.0, .15, .25), oversample_factor=3):
        self.width = width
        self.iteration_limit = iteration_limit
        self.coordinates = coordinates
        self.use_gpu = use_gpu
        self.cycle_length = cycle_length
        self.oversample_factor = oversample_factor
        self.angle_offsets = angle_offsets
        self.height = round(self.width / (self.coordinates[1] - self.coordinates[0]) * (self.coordinates[3] - self.coordinates[2]))
        self.color_palette = create_color_palette(self.angle_offsets)
        self.refresh_fractal()

    # Function to apply zoom at a specific point in the fractal.
    def zoom_focus(self, focus_x, focus_y, scale_factor):
        x_range = (self.coordinates[1] - self.coordinates[0]) / 2
        y_range = (self.coordinates[3] - self.coordinates[2]) / 2

        # Calculate new coordinates centered on the focus point.
        focus_x = focus_x * (1 - scale_factor**2) + (self.coordinates[1] + self.coordinates[0])/2 * scale_factor**2
        focus_y = focus_y * (1 - scale_factor**2) + (self.coordinates[3] + self.coordinates[2])/2 * scale_factor**2
        self.coordinates = [focus_x - x_range * scale_factor,
                            focus_x + x_range * scale_factor,
                            focus_y - y_range * scale_factor,
                            focus_y + y_range * scale_factor]

    # Function to update the fractal set with current settings.
    def refresh_fractal(self):
        # Adjust color cycle and pixel dimensions based on settings.
        color_cycle_adjustment = math.sqrt(self.cycle_length)
        pixel_width = self.width * self.oversample_factor
        pixel_height = self.height * self.oversample_factor

        # Generate the fractal using either GPU or CPU.
        if self.use_gpu:
            self.fractal_image = np.zeros((pixel_height, pixel_width, 3))
            total_pixels = pixel_width * pixel_height
            num_threads = 32
            num_blocks = math.ceil(total_pixels / num_threads)
            generate_fractal_gpu[num_blocks, num_threads](self.fractal_image, *self.coordinates, self.iteration_limit, self.color_palette, color_cycle_adjustment)
        else:
            real_values = np.linspace(self.coordinates[0], self.coordinates[1], pixel_width)
            imag_values = np.linspace(self.coordinates[2], self.coordinates[3], pixel_height)
            self.fractal_image = generate_fractal(real_values, imag_values, self.iteration_limit, self.color_palette, color_cycle_adjustment)

        # Convert the fractal matrix to an image format and apply oversampling if necessary.
        self.fractal_image = (255 * self.fractal_image).astype(np.uint8)
        if self.oversample_factor > 1:
            self.fractal_image = (self.fractal_image
                                  .reshape((self.height, self.oversample_factor,
                                            self.width, self.oversample_factor, 3))
                                  .mean(3).mean(1).astype(np.uint8))

    # Function to create an animation by zooming into the fractal.
    def animate(self, focus_x, focus_y, output_file, frame_count=150, loop=True):
        # Function to generate a Gaussian curve for smooth zoom.
        def gaussian_curve(length, sigma=1):
            x_values = np.linspace(-1, 1, length)
            return np.exp(-np.power(x_values, 2.) / (2 * np.power(sigma, 2.)))
        
        # Calculate the scale factors for each frame.
        scale_factors = 1 - gaussian_curve(frame_count, 1/2) * .3
       
        # Generate the initial fractal image.
        self.refresh_fractal()
        frames = [self.fractal_image]
        for i in range(1, frame_count):
            # Apply zoom for each frame and refresh the fractal.
            self.zoom_focus(focus_x, focus_y, scale_factors[i])
            self.refresh_fractal()
            frames.append(self.fractal_image)
           
        # If loop is enabled, append reversed frames for a seamless loop.
        if loop:
            frames += frames[::-2]

        # Save the animation to a GIF file.
        imageio.mimsave(output_file, frames)  

# Main execution block.
if __name__ == "__main__":
    # Coordinates to zoom into.
    zoom_x_real = -1.749705768080503
    zoom_x_imag = -6.13369029080495e-05
    
    fractal_gpu = FractalExplorer(iteration_limit = 500, width = 426, use_gpu=True)
    fractal_gpu.animate(zoom_x_real, zoom_x_imag, 'mandelbrot.gif')

    # fractal_cpu = FractalExplorer(iteration_limit = 500, width = 426, use_gpu=False)
    # fractal_cpu.animate(zoom_x_real, zoom_x_imag, 'mandelbrot.gif')
