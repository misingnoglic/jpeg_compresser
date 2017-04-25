import numpy as np
import sys
import math
import gzip
import zlib
import bz2
import argparse
import time
import os
import csv
import lzma

from PIL import Image

# Code to make sure py3.6 is used
try:
    assert sys.version_info >= (3, 6)
except:
    raise ValueError("Need to use Python 3.6")


def quantize(d: np.ndarray, q: np.ndarray):
    """
    Divides the matrix d by the quantization matrix q
    :param d: 8x8 or 16x16 matrix
    :param q: quantization matrix
    :return: d/q
    """

    # Rounds each number to nearest number, and then converts to int
    return np.int32(np.matrix.round(d/q))


def dct_transform(a: np.ndarray):
    """
    Given a matrix, does a DCT transform on the matrix
    :param a: matrix being transformed
    :return: the DCT of that matrix
    """
    size = a.shape[0]
    d = gen_dct_matrix(size) # Generate the DCT matrix
    return np.float64(d @ a @ d.T)


def undo_dct(a: np.ndarray):
    """
    Opposite of dct_transform
    :param a: the matrix
    :return: The matrix with the DCT undone
    """
    size = a.shape[0]
    d = gen_dct_matrix(size)
    return np.float64(d.T @ a @ d)


def gen_dct_matrix(size: int):
    """
    Generates the DCT Matrix for multiplication
    Equation for matrix from this website:
    https://www.math.cuhk.edu.hk/~lmlui/dct.pdf
    
    But I wrote the code for it
    
    :param size: Size of the matrix (either 8 or 16)
    :return: Matrix for use in DCT equation
    """

    # First row just 1/sqrt(size)
    dct = np.array([1/math.sqrt(size) for i in range(size)])

    # Each extra row
    for row in range(1,size):
        # Add the new row
        dct = np.vstack([dct, [
            math.sqrt(2/size)*math.cos(
                ((2*j+1) * math.pi * row)/(2*size))
        for j in range(size)]])
    return dct


def crop(im: np.ndarray, block_size: int):
    """
    Crops im so that it fits in the block size (both the height and width)
    :param im: 
    :param block_size: 
    :return: cropped matrix
    """
    shape = im.shape
    new_height = (shape[0] // block_size) * block_size # Taking off the extra bytes
    new_width = (shape[1] // block_size) * block_size
    return im[0:new_height, 0:new_width]


def gen_quant_matrix(size: int = 8, Q=50):
    """
    Generates the quantization matrix. If size = 8, it's the regular matrix.
    If size = 16, then the matrix is 4 blocks of the 8x8 matrix
    :param size: Size of matrix
    :param Q: The quality factor
    :return: 
    """

    quantization_matrix_string = (
        """16  11  10  16  24  40  51  61
        12  12  14  19  26  58  60 55
        14  13  16  24  40  57  69 56
        14  17  22  29  51  87  80 62
        18  22  37  56  68 109 103 77
        24  35  55  64  81 104 113 92
        49  64  78  87 103 121 120 101
        72  92  95  98 112 100 103 99""")

    # Ugly parsing of above string
    int_mat = [[int(y) for y in x.split()] for x in quantization_matrix_string.splitlines()]

    quant_matrix_small = np.int32(int_mat)
    if size == 8: # Possibly this if statement isn't needed, but it works
        m = quant_matrix_small
    else:  # size is 16
        quant_matrix = np.zeros([size,size])
        for i in range(0,size,8): # For each block of 8, copy the small matrix into it
            for j in range(0,size,8):
                quant_matrix[i:i+8, j:j+8] = quant_matrix_small
        m = quant_matrix

    # Changing quality factor of matrix
    # from http://stackoverflow.com/questions/29215879/how-can-i-generalize-the-quantization-matrix-in-jpeg-compression
    if Q < 50:
        s = 5000 / Q
    else:
        s = 200 - 2 * Q

    return np.int32((s * m + 50) / 100)


def zigzag(m: np.int32, odd=True, exclude_dc=False):
    """
    Zigzag algorithm from course slides, with added parameters
    http://www.cs.brandeis.edu/~storer/cs175/Slides/05-JPEG/05-01-JPEG-SLIDES.pdf
    :param m: 8x8 or 16x16 block to zig zag
    :param odd: True if odd traversal, False if even traversal
    :param exclude_dc: True if you exclude the first element
    :return: List of items in zig zag order
    """
    z = []
    n = m.shape[0]
    start = 0
    if exclude_dc:
        start = 1

    for i in range(start,2*(n-1)+1):
        if (i<n):
            bound = 0
        else:
            bound = i-n+1
        for j in range(bound, i-bound+1):
            if (i % 2 == 1) == odd:  # if odd is true,
                z.append(m[j][i-j])
            else:
                z.append(m[i-j][j])
    return z


def unzigzag_matrix(size, odd=True, exclude_dc=False):
    """
    precomputing the un-zig-zag matrix
    :param size: 8 or 16
    :param odd: Odd or even traversal
    :param exclude_dc - whether the DC component is excluded
    """
    m = np.int32(np.zeros([size,size]))
    n = size
    index = -1
    start = 0
    if exclude_dc:
        start = 1

    for i in range(start,2*(n-1)+1):
        if i < n:
            bound = 0
        else:
            bound = i-n+1
        for j in range(bound, i-bound+1):
            index += 1
            # Store the index of the list in the correct part of the matric
            if (i % 2 == 1) == odd:
                m[j][i-j] = index
            else:
                m[i-j][j] = index
    return m


def dc_differences(L):
    """
    Given a list, gives back a list of the first one, plus all the differences  
    L[i]-L[i-1] for each 0<i<len(L) - done because these DC coefficients are
    very highly correlated. Unused in my implementation.
    :param L: List of dc coefficients
    :return: Differences of coefficients
    """
    return [L[0]] + [L[i]-L[i-1] for i in range(1,len(L))]


def undo_dc_differences(L):
    """
    Undoing the differences done in dc_differences. Unsed in this implementation
    :param L: Differences of coefficients
    :return: List of dc coefficients
    """
    for i in range(1,len(L)):
        L[i] = L[i] + L[i-1]
    return L


def unzig_list(m: np.ndarray, L, unzig=None):
    """
    Restores the original shape of a block from the list of items
    created with the zig-zag algorithm
    :param m: The matrix to place into
    :param L: List of items that were zigzagged
    :param unzig: Precomputed matrix for index of each item (like hash table)
    :return: matrix with items in it
    """
    if unzig is None:
        unzig = unzigzag_matrix(m.shape[0])
    height, width = m.shape
    for i in range(height):
        for j in range(width):
            try:
                m[i][j] = L[unzig[i][j]]
            except IndexError:
                print("You must have passed some invalid length list to the method")
                print(f"Should be length {max(unzig)}, yours is length {len(L)}.")
                raise
    return m

# Defining the various ways to compress/decompress
# Will be added to the map below, so the functions can choose the right one
# Here are all the python compression libraries:
# https://docs.python.org/3/library/archiving.html

# The compress functions all take a list of bytes, and a filename to store them
# The decompress functions all take the filename, and return the list of bytes


def gzip_compress(list_of_bytes, filename):
    with gzip.open(filename, 'wb') as f:
        f.write(list_of_bytes)


def gzip_uncompress(filename):
    with gzip.open(filename, 'rb') as f:
        zags = list(f.read())
    return zags


def zlib_compress(list_of_bytes, filename):
    with open(filename, 'wb') as f:
        z = zlib.compress(list_of_bytes)
        f.write(z)


def zlib_uncompress(filename):
    with open(filename, 'rb') as f:
        r = f.read()
        z = zlib.decompress(r)
    return z


def bzip_compress(list_of_bytes, filename):
    with open(filename, 'wb') as f:
        z = bz2.compress(list_of_bytes)
        f.write(z)


def bzip_uncompress(filename):
    with open(filename, 'rb') as f:
        r = f.read()
        z = bz2.decompress(r)
    return z

def xz_compress(list_of_bytes, filename):
    with open(filename, 'wb') as f:
        z = lzma.compress(list_of_bytes)
        f.write(z)

def xz_uncompress(filename):
    with open(filename, 'rb') as f:
        r = f.read()
        z = lzma.decompress(r)
    return z



# Map of compression type to the compression and decompression methods
compression_map = {"gzip": (gzip_compress, gzip_uncompress),
                   "zlib": (zlib_compress, zlib_uncompress),
                   "bzip": (bzip_compress, bzip_uncompress),
                   "xz": (xz_compress, xz_uncompress)}

# Key for encoding the compression type - unused
compression_key = {1: "gzip", 2: "zlib", 3: "bzip"}


def encode_length(n, block_size):
    """"
    Encoding the height and width in 2 bytes
    First byte is n//(block_size * 255), and then second is the leftover divided by block size.
    For block size 8, this allows images 522240x522240
    For block size 16, this allows images 1044480x1044480
    
    This works because the height and the width of the image are always divisible by the 
    """

    high_byte_value = block_size * 255
    # Return tuple of high byte, low byte
    return n//high_byte_value, (n%high_byte_value)//block_size


def decode_length(high, low, block_size):
    """
    Undoes the 'encode_length' algorithm from above
    :param high: the high bit
    :param low: the low bit
    :param block_size: the block size
    :return: the actual length
    """
    return (high * (block_size * 255)) + low * block_size


def matrix_to_zags(im, block_size, quality_factor):
    """
    Turns a matrix into a list of items - using the zig-zag algorithm in class on each block in the matrix
    But first applying the dct transformation to the block, and then quantizing it. 
    :param im: matrix to zig-zag
    :param block_size: block size for doing the zig-zag
    :param quality_factor: quality factor for creating the quantization matrix
    :return: the new matrix, the dc coefficients, and the list of items in the zig-zag
    """
    height, width = im.shape
    dc_coefficients = []
    zags = []
    # create the quantization matrix
    quant_matrix = gen_quant_matrix(block_size, quality_factor)

    # for each block
    for h in range(0, height, block_size):
        for w in range(0, width, block_size):
            # Do the dct transformation on that block, and then quantize it
            im[h: h+block_size, w:w+block_size] = quantize(dct_transform(im[h: h+block_size, w:w+block_size]), quant_matrix)

            # Add the DC coefficient to the list of DC coefficients
            dc_coefficients.append(im[h][w])

            # add the zig zags from that block to the large list of them
            zags.extend(zigzag(im[h: h+block_size, w:w+block_size], exclude_dc=True))
    return im, dc_coefficients, zags


def compress_color(input_filename, output_filename, block_size, quality_factor, compression_method, verbose):
    """
    Main compression algorithm for color pics
    :param input_filename: name of bitmap
    :param output_filename: name of eventual compressed file
    :param block_size: block size used for applying DCT
    :param quality_factor: quality factor for quantization matrix
    :param compression_method: lossless compression method
    :param verbose: whether to print
    :return: compression ratio
    """
    start_time = time.time()
    # splitting the channels
    ims = [np.float32(x) for x in Image.open(input_filename).split()]
    if verbose:
        print(f"Opened color image {input_filename} for compression, split into 3 channels")

    # Step 0 - crop image in bottom right
    old_shape = ims[0].shape
    ims = [crop(im, block_size) for im in ims]
    if verbose and not ims[0].shape==old_shape:
        print(f"Cropped image from {old_shape[0]}x{old_shape[1]} to {im.shape[0]}x{im.shape[1]} to "
              f"fit the block size {block_size}")

    # Step 1, subtract 128 from each block in the matrix:
    ims = [np.int32(im) - 128 for im in ims]
    if verbose:
        print("Leveled each block by subtracting 128")

    # Step 2, perform DCT on each block of the image, collecting the zig zags of each block
    zags = []
    dc_coefficients = []
    height, width = ims[0].shape
    for level in range(3):
        im, temp_dc_coefficients, temp_zags = matrix_to_zags(ims[level], block_size, quality_factor)
        ims[level] = im
        zags.extend(temp_zags)
        dc_coefficients.extend(temp_dc_coefficients)

    if verbose:
        print(f"Performed DCT on each block of the image, and zig-zagged the blocks into a list")

    # dc_coefficients = dc_differences(dc_coefficients)
    zags = dc_coefficients + zags

    if max(width,height)>(255*block_size)*256:
        raise ValueError("You need a bigger block size to compress this image")

    # Encoding the height and width
    height_high, height_low = encode_length(height, block_size)
    width_high, width_low = encode_length(width, block_size)

    zags.extend([width_low, width_high, height_low, height_high, quality_factor, block_size])
    if verbose:
        print("Encoded the width, height, block size, and compression method")

    # We need to add m to each byte, because the bytes() function in python can only take lists
    # of numbers between 0-255
    m = min(zags)
    try:
        b = bytes([x + (-m) for x in zags] + [-m])
    except ValueError:
        print("Your quality factor must be too high. Try a lower one. ")
        raise

    # Get the right compression function
    compression_function = compression_map[compression_method][0]

    compression_function(b, output_filename)
    old_size = os.stat(input_filename).st_size
    new_size = os.stat(output_filename).st_size

    if verbose:
        print(f"Compressed {input_filename} to {output_filename} using lossless compression method {compression_method}")
        print(f"Block size {block_size} and quality factor {quality_factor}")
        print(f"Took {round(time.time() - start_time, 2)} seconds")
        old_size = os.stat(input_filename).st_size
        new_size = os.stat(output_filename).st_size
        print(f"Original file size: {old_size} bytes")
        print(f"New file size: {new_size} bytes")
        print(f"Compression Ratio: {round(old_size/new_size, 3)}")

    return round(old_size / new_size, 3)


def compress_bw(input_filename, output_filename, block_size, quality_factor, compression_method, verbose):
    """
    Main compression algorithm for b&w pics

    :param input_filename: name of bitmap
    :param output_filename: name of eventual compressed file
    :param block_size: block size used for applying DCT
    :param quality_factor: quality factor for quantization matrix
    :param compression_method: lossless compression method
    :param verbose: whether to print
    :return: compression ratio
    """

    start_time = time.time()
    im = np.float32(Image.open(input_filename))
    if verbose:
        print(f"Opened file {input_filename} for compression")



    # Step 0 - crop image in bottom right
    old_shape = im.shape
    im = crop(im, block_size)
    if verbose and not im.shape == old_shape:
        print(f"Cropped image from {old_shape[0]}x{old_shape[1]} to {im.shape[0]}x{im.shape[1]} to "
              f"fit the block size {block_size}")

    # Step 1, subtract 128 from each block in the matrix:
    im = np.int32(im) - 128
    if verbose:
        print("Leveled each block by subtracting 128")

    # Step 2, perform DCT on each block of the image, collecting the zig zags of each block
    height, width = im.shape

    im, dc_coefficients, zags = matrix_to_zags(im, block_size, quality_factor)
    # dc_coefficients = dc_differences(dc_coefficients)
    dc_coefficients = [x for x in dc_coefficients]

    zags = dc_coefficients + zags

    if verbose:
        print("Performed DCT on each block of the image, and zig-zagged the blocks into a list")

    if max(width,height) > (255*block_size)*256:
        raise ValueError("You need a bigger block size to compress this image, because"
                         " of how we're encoding the height and width.")

    # Encoding the height and width
    height_high, height_low = encode_length(height, block_size)
    width_high, width_low = encode_length(width, block_size)

    zags.extend([width_low, width_high, height_low, height_high, quality_factor, block_size])
    if verbose:
        print("Encoded the width, height, block size, and compression method")

    # We need to add 128 to each byte, because the bytes() function in python can only take lists
    # of numbers between 0-255
    m = min(zags)
    try:
        b = bytes([x + (-m) for x in zags] + [-m])
    except ValueError:
        print("Your quality factor must be too high. Try a lower one. ")
        raise

    # Get the right compression function
    compression_function = compression_map[compression_method][0]

    # call it on the bytes
    compression_function(b, output_filename)

    old_size = os.stat(input_filename).st_size
    new_size = os.stat(output_filename).st_size

    if verbose:
        print(f"Compressed {input_filename} to {output_filename} using lossless compression method {compression_method}")
        print(f"Block size {block_size} and quality factor {quality_factor}")
        print(f"Took {round(time.time() - start_time, 2)} seconds")
        print(f"Original file size: {old_size} bytes")
        print(f"New file size: {new_size} bytes")
        print(f"Compression Ratio: {round(old_size/new_size, 3)}")

    return round(old_size/new_size, 3)


def compress(input_filename, output_filename, block_size, quality_factor, compression_method, verbose):
    """
    Takes in the image and decides whether to send it into the color compressor or the b&w compressor
    :param input_filename: name of bitmap
    :param output_filename: name of eventual compressed file
    :param block_size: block size used for applying DCT
    :param quality_factor: quality factor for quantization matrix
    :param compression_method: lossless compression method
    :param verbose: whether to print
    :return: compression ratio
    """

    x = np.int32(Image.open(input_filename))
    if len(x.shape) > 2: # If the matrix has 3 dimensions, then it's a color image
        if verbose:
            print("Image is color image. ")
        ratio = compress_color(input_filename, output_filename, block_size, quality_factor, compression_method, verbose)
    else:
        if verbose:
            print("Image is greyscale image. ")
        ratio = compress_bw(input_filename, output_filename, block_size, quality_factor, compression_method, verbose)
    return ratio



def view_jpg_file_bw(zags, height, width, quality_factor, start_time, block_size, verbose):
    """
    Views the grayscale JPEG image
    :param zags: List of the zig zags from the image
    :param height: height of image
    :param width: width of image
    :param quality_factor: quality factor used for quantization matrix
    :param start_time: time it took to start the viewing
    :param block_size: block size used
    :param verbose: whether to print
    """

    dc_coefficients = zags[:(height*width)//(block_size**2)]
    zags = zags[(height*width)//(block_size**2):]

    im = np.float32(np.zeros([height, width]))
    uz = unzigzag_matrix(block_size)

    current_index = 0
    dc_index = 0
    square_size = block_size**2

    quant_matrix = gen_quant_matrix(block_size, quality_factor)

    for h in range(0, height, block_size):
        for w in range(0, width, block_size):
            new_block = np.float32(np.zeros([block_size, block_size]))
            new_block = unzig_list(new_block, [dc_coefficients[dc_index]] + zags[current_index:current_index + square_size], uz)
            new_block = new_block * quant_matrix
            new_block = undo_dct(new_block)

            im[h: h+block_size, w:w+block_size] = new_block
            current_index += (block_size**2 -1)
            dc_index += 1
    if verbose:
        print("Restored each block by un-zigzagging, multiplying by the quantization matrix, and undoing the DCT")

    im = np.int32(np.matrix.round(im)) + 128
    if verbose:
        print("Re-leveled image by adding 128 to each pixel")

    pillow_image = Image.fromarray(im)
    if verbose:
        print("Displaying Image (check your task bar)")
        print(f"Time taken: {round(time.time()-start_time, 2)} seconds")
    pillow_image.show()

def view_jpg_file_color(zags, height, width, quality_factor, start_time, block_size, verbose):
    """
    Views the color JPEG image
    :param zags: List of the zig zags from the image
    :param height: height of image
    :param width: width of image
    :param quality_factor: quality factor used for quantization matrix
    :param start_time: time it took to start the viewing
    :param block_size: block size used
    :param verbose: whether to print
    """
    dc_end = 3*((height * width) // (block_size ** 2))
    dc_coefficients = zags[:dc_end]
    #dc_coefficients = undo_dc_differences(zags[:dc_end])
    zags = zags[dc_end:]

    im = np.float32(np.zeros([height, width, 3]))
    uz = unzigzag_matrix(block_size)

    current_index = 0
    dc_index = 0
    square_size = block_size**2

    quant_matrix = gen_quant_matrix(block_size, quality_factor)

    for level in range(3):
        for h in range(0, height, block_size):
            for w in range(0, width, block_size):
                new_block = np.float32(np.zeros([block_size, block_size]))
                new_block = unzig_list(new_block, [dc_coefficients[dc_index]] + zags[current_index:current_index + square_size], uz)
                new_block = new_block * quant_matrix
                new_block = undo_dct(new_block)

                im[h: h+block_size, w:w+block_size, level] = new_block
                current_index += block_size**2 - 1
                dc_index += 1

    if verbose:
        print("Restored each block by un-zigzagging, multiplying by the quantization matrix, and undoing the DCT")

    # Saturate
    im = np.matrix.round(im +128)
    im[im < 0] = 0
    im[im > 255] = 255
    im = np.uint8(im)

    if verbose:
        print("Re-leveled image by adding 128 to each pixel")

    pillow_image = Image.fromarray(im, mode="RGB")
    if verbose:
        print("Displaying Image (check your task bar)")
        print(f"Time taken: {round(time.time()-start_time, 2)} seconds")
    pillow_image.show()


def view_jpg_file(filename, compression_method, verbose):
    """
    Method that takes in the compressed file, and passes it to the appropriate viewer (either color or b&w)
    :param filename: Name of compressed file
    :param compression_method: Compression method used
    :param verbose: Whether to print debug info or not
    """

    start_time = time.time()
    file_name = filename

    # Get the right decompression function
    decompression_function = compression_map[compression_method][1]
    zags = list(decompression_function(file_name))

    if verbose:
        print(f"Decompressed {filename} using lossless decompression method {compression_method}")

    # subtract m from each block, since we add m before compressing
    m = zags.pop()
    zags = [x - m for x in zags]

    block_size = zags.pop()
    quality_factor = zags.pop()

    # Unpacking the width and height of the image
    height_high = zags.pop()
    height_low = zags.pop()

    width_high = zags.pop()
    width_low = zags.pop()

    height = decode_length(height_high, height_low, block_size)
    width = decode_length(width_high, width_low, block_size)

    if height*width < len(zags):
        if verbose: print("This is a color photo")
        view_jpg_file_color(zags, height, width, quality_factor, start_time, block_size, verbose)
    else:
        if verbose: print("This is a grayscale photo")
        view_jpg_file_bw(zags, height, width, quality_factor, start_time, block_size, verbose)

# Functions made for testing:


def checkerboard(height: int, width: int):
    """
    Creates the checkerboard example from the slides, but can be
    arranged to any size. In the slides it was 8x8
    :param height: height of checkerboard
    :param width: width of checkerboard
    :return: Checkerboard matrix
    """
    # Test from slides - checker board
    x = [[50] * (width // 2) + [200] * (width // 2) for x in range(height // 2)]
    x += [list(reversed(p)) for p in x]
    im = np.float32(x)
    return im


def color_checkerboard(height: int, width: int):
    """
    Color checkerboard (similar to the b&w one from above)    
    """
    x = [[[50,50,50]] * (width // 2) + [[200, 200, 200]] * (width // 2) for x in range(height // 2)]
    x += [list(reversed(p)) for p in x]
    im = np.float32(x)
    return im


def zig_zag_example():
    """
    Code used for testing - returns the 8x8 matrix which is 
    used as an example in Storer's slides for testing 
    the zig zag algorithm
    :return: zig zag matrix
    """
    s = np.zeroes([8,8])
    for i in range(64):
        s[i//8][i%8] = i

    return np.int32(s)


def zig_zag_test():
    """
    Function for testing the zig-zag algorithm
    """
    z = zig_zag_example()
    print(z)
    print()
    p = zigzag(zig_zag_example())
    uz = unzigzag_matrix(8)
    print(uz)
    print()
    im = np.zeros([8,8])
    r = unzig_list(im, p, uz)
    print(r)


def compress_all():
    """
    Compresses all of the bmp files in your folder, and creates a CSV report with all the data.
    Made to create the table of data in the PDF report. 
    """
    import glob
    bmps = glob.glob("*.bmp")
    chart = csv.writer(open("report.csv", "w"), lineterminator='\n')
    # Map my quality to PIL quality
    quality_map = {50: 95, 40: 75, 30: 35}

    for quality in [50,40,30]:
        stuff = []
        for bmp in bmps:
            L = []
            L.append(bmp)
            old_size = os.stat(bmp).st_size
            L.append(old_size)
            # Finding the regular compression type by using PIL to create a JPG
            im = Image.open(bmp)
            im.save("test.jpg", format="JPEG", quality = quality_map[quality])

            new_size = os.stat("test.jpg").st_size
            L.append(round(old_size/new_size, 3))

            for compression_type in ["gzip", "bzip", "zlib", "xz"]:
                for block_size in [8,16]:
                    new = bmp.replace("bmp", compression_type)
                    ratio = compress(bmp, new, block_size,
                             quality, compression_type, False)
                    L.append(ratio)
                    #view_jpg_file(new, compression_type, False)
                    #input()
            chart.writerow(L)


def main():
    """
    Main Method: Generates the command line arguments, and handles the interactive version, as well
    as designates which methods be called
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--verbose", help="Prints Messages", action="store_true")
    parser.add_argument("--interactive", help="Makes program interactive", action="store_true")
    parser.add_argument("--action", help="Program Action", choices = ["compress", "decompress", "both"])
    parser.add_argument("--originalfile",
                        help="Filename for compressed file(either for storing the compressed file or reading it. ")
    parser.add_argument("--compressedfile",
                        help="Filename for compressed file(either for storing the compressed file or reading it. ")
    parser.add_argument("--blocksize", type=int, help="Block size for image (for compression)", choices=[8,16])
    parser.add_argument("--qualityfactor", type=int, help="Quality Factor (generally 30,40, 50)")
    parser.add_argument("--compressionmethod", help="Compression Method", choices=["gzip", "zlib", "bzip", "xz"])

    args = parser.parse_args()

    if args.verbose:
        print("Starting program with these arguments: ")
        print(args)

    if args.interactive:
        print("Please note there is no error checking in this interactive version, so don't type things wrong.")
        action = input("Would you like to [compress], [decompress], or [both]: ")

        if action == "compress" or action == "both":
            og_file = input("Type the original file name: ")
            bs = int(input("Please type the blocksize (8 or 16): "))
            qf = int(input("Please type the quality factor (30, 40, or 50): "))

        comp_file = input("Type the name for the compressed file: ")
        cm = input("Type the compression method [bzip/gzip/zlib/xz]: ")
        vb = input("Would you like your program to report what it's doing? [Y/N]: ").lower() == "y"
    else:
        # If not interactive, just get the arguments from the command line
        action = args.action
        og_file = args.originalfile
        comp_file = args.compressedfile
        bs = args.blocksize
        qf = args.qualityfactor
        cm = args.compressionmethod
        vb = args.verbose

    if action == "compress" or action == "both":
        compress(og_file, comp_file, bs,
                    qf, cm, vb)
        print()

    if action == "decompress" or action == "both":
        view_jpg_file(comp_file, cm, vb)


if __name__=="__main__":
    main()