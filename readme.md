Arya Boudaie

COSI 175A

# Jpeg Project Description

This is the description of my program - how to use it and how it works. Keep in mind that it works on both color and grayscale bitmaps. To use it, you need to install the numpy and Pillow libraries, and you need to run it on Python 3.6.

# How to use

The program, jpeg_compressor.py, is a python program that runs with command line arguments. If you don’t want to deal with command line arguments, you can just pass the --interactive argument to have the program prompt you. You can run *jpeg_compressor.py -h* for all the arguments. The most important one is --action {compress,decompress,both}

If your action is "compress" or “both,” you need to have these options:

  --originalfile ORIGINALFILE

                        Filename for compressed file(either for storing the

                        compressed file or reading it.

  --compressedfile COMPRESSEDFILE

                        Filename for compressed file(either for storing the

                        compressed file or reading it.

  --blocksize BLOCKSIZE

                        Block size for image (for compression)

  --qualityfactor QUALITYFACTOR

                        Quality Factor (generally 30,40, 50)

  --compressionmethod {gzip,zlib,bzip,xz}

                        Compression Method

If your action is decompress, you only need to supply the "compressedfile" and “compressionmethod” arguments. For all actions, you can also supply this argument.

  --verbose             Prints Messages

If you want the program to annotate what it’s doing, plus tell you the compression ratios. For example:

python jpeg_compressor.py --action both --verbose --originalfile Kodak12gray.bmp --compressedfile file.gz --blocksize 8 --qualityfactor 50 --compressionmethod gzip

This command will first compress "Kodak12gray.bmp" into file.gz, with a blocksize of 8, quality factor of 50, and compress with gzip. It will then uncompress file.gz with gzip, and display the image. Because --verbose is turned on, the program will also print out what it’s doing.

An example of a decompress call to the above image:

python jpeg_compressor.py --action decompress --verbose --compressedfile file.gz --compressionmethod gzip

Notice how for the decompression, you just have to specify the filename and the compression method, since information like the block size is encoded in the image.

# How the Program Works:

The code is fairly heavily commented, so I will just give a brief overview of how my program works. The main data structures used are the numpy matrices, and the PIL image format, which is used for reading and writing images.

To compress, the program first reads in the image as a numpy matrix, and then decides whether to send it to the color compressor or the grayscale compressor. These are the steps the compressor takes:

1. Read the image as a matrix

2. Crop the image to the block size

3. Level the matrix by subtracting 128

4. On each block of the matrix:

    1. Perform the DCT (generate DCT matrix, do dct_matrix * block * transform_dct_matrix)

    2. Quantize it (using the quantization matrix from the standard, modified with the quality factor).

    3. Get all of the items in the zig-zag order and add them to a list

    4. Add the DC component to a list of DC component

5. Add the DC components in front of the zig_zag list

6. Encode the height and width of the image into two bytes, using the formula:
high_byte = n/(block_size * 255)
low_byte = (n%(block_size * 255))/block_size

7. Add the width, height,quality factor, and block size to the zig zag list, so that it is encoded into the image (so the user doesn’t have to know these things).

8. Add a certain number m to each item in the list, so all of the bytes are above 0. Then, add m.

9. Compress the list of bytes using the lossless compression given by the user (gzip, bzip, zlib).

10. Save the compressed version as a file, and report statistics.

If the image is color, steps 2-4 are done on all 3 channels, and added to one long list of zags.

To view the compressed image:

1. Uncompress the file using the compression method given by the user.

2. Get the m, and subtract it from each item.

3. Get all of the relevant metadata (blocksize, quality factor, height, width), and use the height and width to find out if the image is color or b&w

4. Separate the DC components from the AC components

5. Create a new matrix that is height x width (if it is color, make height x width x 3)

6. For each section of the list that is block_size^2:

    1. Restore the original shape (undoing the zig-zag algorithm and adding the DC component for that block)

    2. Multiply by the quantization matrix

    3. Undo the DCT (dctmatrix transform * m * dctmatrix)

    4. Add it to the appropriate part of the new matrix

7. Re-level the matrix by adding 128

8. Display the image as a matrix

    5. If the image is a color image, it needs to be saturated (anything less than 0 = 0, anything greater than 255 = 255), and then turn it into an uint8 matrix

9. Display statistics (how long it took).

Note that for step 6, if the image is color, it will fill the blocks into all 3 channels.

One of the benefits of my style with lots of functions is that it is very easy to plug and change certain parts. For example, to add a new lossless compression, you just have to define the compress/uncompress functions, add it to the dictionary of compression methods, and add it as a command line argument. I added xz at the last minute due to a friend’s advice.

Some things I wish I did better:

* Be able to encode the DC differences as opposed to just the DC coefficients - when I did it, I just got a max-min greater than 256, so I couldn’t encode it as a byte.

* Avoid having to add that ‘m’ and subtract it later.

* Somehow encode the compression technique so that the user does not have to enter it. The problem with this is that the bytes become compressed, so I wouldn’t know what to decompress it with to get the data.

# Example run:

>python jpeg_compressor.py --action both --verbose --originalfile Kodak12.bmp --compressedfile file.gz --blocksize 8 --qualityfactor 50 --compressionmethod gzip

Starting program with these arguments:

Namespace(action='both', blocksize=8, compressedfile='file.gz', compressionmethod='gzip', interactive=False, originalfile='Kodak12.bmp', qualityfactor=50, verbose=True)

Image is color image.

Opened color image Kodak12.bmp for compression, split into 3 channels

Leveled each block by subtracting 128

Performed DCT on each block of the image, and zig-zagged the blocks into a list

Encoded the width, height, block size, and compression method

Compressed Kodak12.bmp to file.gz using lossless compression method gzip

Block size 8 and quality factor 50

Took 4.33 seconds

Original file size: 1179704 bytes

New file size: 110256 bytes

Compression Ratio: 10.7

Decompressed file.gz using lossless decompression method gzip

This is a color photo

Restored each block by un-zigzagging, multiplying by the quantization matrix, and undoing the DCT

Re-leveled image by adding 128 to each pixel

Displaying Image (check your task bar)

Time taken: 3.76 seconds

# Tables

The following is a result of my compression algorithm on the test images. For each one, I include the filename, original size (in bytes), compression ratio achieved with Pillow’s JPEG converter (with quality  = 95 for PSNR 50, q=75 for 40, and q=35 for 30), and the compression ratio received with the blocksize and lossless compression technique. The block sizes I used were 8 and 16, and the lossless compressions I used were gzip, bzip, zlib, and xz.

## PSNR = 50

<table>
  <tr>
    <td>Filename</td>
    <td>Og. Size (bytes)</td>
    <td>JPEG Compression ratio</td>
    <td>Ratio (8/gzip)</td>
    <td>16/gzip</td>
    <td>8/bzip</td>
    <td>16/bzip</td>
    <td>8/zlib</td>
    <td>16/zlib</td>
    <td>8/xz</td>
    <td>16/xz</td>
  </tr>
  <tr>
    <td>Kodak08.bmp</td>
    <td>1179704</td>
    <td>5.089</td>
    <td>4.945</td>
    <td>4.992</td>
    <td>5.983</td>
    <td>6.406</td>
    <td>4.787</td>
    <td>4.729</td>
    <td>8.492</td>
    <td>9.039</td>
  </tr>
  <tr>
    <td>Kodak08gray.bmp</td>
    <td>394296</td>
    <td>1.865</td>
    <td>4.863</td>
    <td>4.921</td>
    <td>5.422</td>
    <td>5.5</td>
    <td>4.72</td>
    <td>4.669</td>
    <td>5.617</td>
    <td>5.508</td>
  </tr>
  <tr>
    <td>Kodak09.bmp</td>
    <td>1179704</td>
    <td>8.873</td>
    <td>11.425</td>
    <td>12.698</td>
    <td>14.409</td>
    <td>16.882</td>
    <td>10.89</td>
    <td>11.852</td>
    <td>17.739</td>
    <td>20.375</td>
  </tr>
  <tr>
    <td>Kodak09gray.bmp</td>
    <td>394296</td>
    <td>3.181</td>
    <td>11.181</td>
    <td>12.337</td>
    <td>12.779</td>
    <td>14.323</td>
    <td>10.67</td>
    <td>11.536</td>
    <td>13.013</td>
    <td>14.12</td>
  </tr>
  <tr>
    <td>Kodak12.bmp</td>
    <td>1179704</td>
    <td>9.005</td>
    <td>10.699</td>
    <td>11.914</td>
    <td>13.83</td>
    <td>16.278</td>
    <td>10.178</td>
    <td>11.064</td>
    <td>17.028</td>
    <td>19.782</td>
  </tr>
  <tr>
    <td>Kodak12gray.bmp</td>
    <td>394296</td>
    <td>3.139</td>
    <td>9.927</td>
    <td>11.116</td>
    <td>11.692</td>
    <td>13.163</td>
    <td>9.498</td>
    <td>10.35</td>
    <td>11.743</td>
    <td>12.76</td>
  </tr>
  <tr>
    <td>Kodak18.bmp</td>
    <td>1179704</td>
    <td>5.783</td>
    <td>6.156</td>
    <td>6.52</td>
    <td>7.472</td>
    <td>8.28</td>
    <td>5.936</td>
    <td>6.142</td>
    <td>9.393</td>
    <td>10.546</td>
  </tr>
  <tr>
    <td>Kodak18gray.bmp</td>
    <td>394296</td>
    <td>2.341</td>
    <td>6.481</td>
    <td>6.997</td>
    <td>7.591</td>
    <td>8.122</td>
    <td>6.256</td>
    <td>6.617</td>
    <td>7.677</td>
    <td>7.982</td>
  </tr>
  <tr>
    <td>Kodak21.bmp</td>
    <td>1179704</td>
    <td>6.964</td>
    <td>7.844</td>
    <td>8.083</td>
    <td>9.798</td>
    <td>10.679</td>
    <td>7.555</td>
    <td>7.609</td>
    <td>13.156</td>
    <td>14.344</td>
  </tr>
  <tr>
    <td>Kodak21gray.bmp</td>
    <td>394296</td>
    <td>2.579</td>
    <td>8.047</td>
    <td>8.271</td>
    <td>9.168</td>
    <td>9.441</td>
    <td>7.778</td>
    <td>7.785</td>
    <td>9.438</td>
    <td>9.45</td>
  </tr>
  <tr>
    <td>Kodak22.bmp</td>
    <td>1179704</td>
    <td>6.776</td>
    <td>7.552</td>
    <td>8.242</td>
    <td>9.104</td>
    <td>10.474</td>
    <td>7.267</td>
    <td>7.702</td>
    <td>11.184</td>
    <td>12.678</td>
  </tr>
  <tr>
    <td>Kodak22gray.bmp</td>
    <td>394296</td>
    <td>2.598</td>
    <td>7.629</td>
    <td>8.416</td>
    <td>8.927</td>
    <td>9.819</td>
    <td>7.35</td>
    <td>7.849</td>
    <td>9.155</td>
    <td>9.502</td>
  </tr>
</table>


## PSNR = 40

<table>
  <tr>
    <td>Filename</td>
    <td>Og. Size (bytes)</td>
    <td>JPEG Compression ratio</td>
    <td>Ratio (8/gzip)</td>
    <td>16/gzip</td>
    <td>8/bzip</td>
    <td>16/bzip</td>
    <td>8/zlib</td>
    <td>16/zlib</td>
    <td>8/xz</td>
    <td>16/xz</td>
  </tr>
  <tr>
    <td>Kodak08.bmp</td>
    <td>1179704</td>
    <td>11.588</td>
    <td>5.588</td>
    <td>5.75</td>
    <td>6.852</td>
    <td>7.503</td>
    <td>5.391</td>
    <td>5.425</td>
    <td>9.742</td>
    <td>10.497</td>
  </tr>
  <tr>
    <td>Kodak08gray.bmp</td>
    <td>394296</td>
    <td>4.159</td>
    <td>5.487</td>
    <td>5.658</td>
    <td>6.138</td>
    <td>6.396</td>
    <td>5.308</td>
    <td>5.354</td>
    <td>6.37</td>
    <td>6.362</td>
  </tr>
  <tr>
    <td>Kodak09.bmp</td>
    <td>1179704</td>
    <td>25.103</td>
    <td>13.26</td>
    <td>14.973</td>
    <td>17</td>
    <td>20.607</td>
    <td>12.669</td>
    <td>13.911</td>
    <td>21.304</td>
    <td>24.522</td>
  </tr>
  <tr>
    <td>Kodak09gray.bmp</td>
    <td>394296</td>
    <td>8.978</td>
    <td>12.891</td>
    <td>14.509</td>
    <td>14.942</td>
    <td>17.181</td>
    <td>12.319</td>
    <td>13.534</td>
    <td>15.077</td>
    <td>16.651</td>
  </tr>
  <tr>
    <td>Kodak12.bmp</td>
    <td>1179704</td>
    <td>23.742</td>
    <td>12.697</td>
    <td>14.424</td>
    <td>16.808</td>
    <td>20.084</td>
    <td>12.038</td>
    <td>13.329</td>
    <td>19.98</td>
    <td>23.848</td>
  </tr>
  <tr>
    <td>Kodak12gray.bmp</td>
    <td>394296</td>
    <td>8.247</td>
    <td>11.758</td>
    <td>13.373</td>
    <td>14.039</td>
    <td>16.047</td>
    <td>11.19</td>
    <td>12.416</td>
    <td>13.779</td>
    <td>15.33</td>
  </tr>
  <tr>
    <td>Kodak18.bmp</td>
    <td>1179704</td>
    <td>14.199</td>
    <td>7.038</td>
    <td>7.69</td>
    <td>8.644</td>
    <td>9.929</td>
    <td>6.769</td>
    <td>7.206</td>
    <td>10.846</td>
    <td>12.466</td>
  </tr>
  <tr>
    <td>Kodak18gray.bmp</td>
    <td>394296</td>
    <td>5.643</td>
    <td>7.42</td>
    <td>8.197</td>
    <td>8.732</td>
    <td>9.639</td>
    <td>7.152</td>
    <td>7.711</td>
    <td>8.836</td>
    <td>9.411</td>
  </tr>
  <tr>
    <td>Kodak21.bmp</td>
    <td>1179704</td>
    <td>18.035</td>
    <td>9.017</td>
    <td>9.436</td>
    <td>11.482</td>
    <td>12.805</td>
    <td>8.674</td>
    <td>8.851</td>
    <td>15.46</td>
    <td>17.001</td>
  </tr>
  <tr>
    <td>Kodak21gray.bmp</td>
    <td>394296</td>
    <td>6.668</td>
    <td>9.202</td>
    <td>9.619</td>
    <td>10.641</td>
    <td>11.223</td>
    <td>8.898</td>
    <td>9.048</td>
    <td>10.928</td>
    <td>11.016</td>
  </tr>
  <tr>
    <td>Kodak22.bmp</td>
    <td>1179704</td>
    <td>17.369</td>
    <td>8.742</td>
    <td>9.81</td>
    <td>10.658</td>
    <td>12.716</td>
    <td>8.371</td>
    <td>9.134</td>
    <td>13.012</td>
    <td>15.05</td>
  </tr>
  <tr>
    <td>Kodak22gray.bmp</td>
    <td>394296</td>
    <td>6.567</td>
    <td>8.861</td>
    <td>10.007</td>
    <td>10.383</td>
    <td>11.942</td>
    <td>8.497</td>
    <td>9.334</td>
    <td>10.593</td>
    <td>11.398</td>
  </tr>
</table>


## PSNR = 30

<table>
  <tr>
    <td>Filename</td>
    <td>Og. Size (bytes)</td>
    <td>JPEG Compression ratio</td>
    <td>Ratio (8/gzip)</td>
    <td>16/gzip</td>
    <td>8/bzip</td>
    <td>16/bzip</td>
    <td>8/zlib</td>
    <td>16/zlib</td>
    <td>8/xz</td>
    <td>16/xz</td>
  </tr>
  <tr>
    <td>Kodak08.bmp</td>
    <td>1179704</td>
    <td>20.975</td>
    <td>6.523</td>
    <td>6.888</td>
    <td>8.164</td>
    <td>9.252</td>
    <td>6.264</td>
    <td>6.482</td>
    <td>11.641</td>
    <td>12.694</td>
  </tr>
  <tr>
    <td>Kodak08gray.bmp</td>
    <td>394296</td>
    <td>7.463</td>
    <td>6.421</td>
    <td>6.772</td>
    <td>7.26</td>
    <td>7.734</td>
    <td>6.18</td>
    <td>6.393</td>
    <td>7.521</td>
    <td>7.617</td>
  </tr>
  <tr>
    <td>Kodak09.bmp</td>
    <td>1179704</td>
    <td>46.697</td>
    <td>15.782</td>
    <td>18.164</td>
    <td>20.887</td>
    <td>25.819</td>
    <td>15.088</td>
    <td>16.835</td>
    <td>26.118</td>
    <td>30.11</td>
  </tr>
  <tr>
    <td>Kodak09gray.bmp</td>
    <td>394296</td>
    <td>17.338</td>
    <td>15.289</td>
    <td>17.578</td>
    <td>17.994</td>
    <td>21.355</td>
    <td>14.694</td>
    <td>16.329</td>
    <td>18.174</td>
    <td>20.337</td>
  </tr>
  <tr>
    <td>Kodak12.bmp</td>
    <td>1179704</td>
    <td>45.594</td>
    <td>15.865</td>
    <td>18.317</td>
    <td>21.881</td>
    <td>26.292</td>
    <td>14.932</td>
    <td>16.883</td>
    <td>24.406</td>
    <td>29.797</td>
  </tr>
  <tr>
    <td>Kodak12gray.bmp</td>
    <td>394296</td>
    <td>16.207</td>
    <td>14.572</td>
    <td>16.913</td>
    <td>17.928</td>
    <td>20.558</td>
    <td>13.759</td>
    <td>15.567</td>
    <td>16.882</td>
    <td>19.309</td>
  </tr>
  <tr>
    <td>Kodak18.bmp</td>
    <td>1179704</td>
    <td>27.137</td>
    <td>8.397</td>
    <td>9.499</td>
    <td>10.471</td>
    <td>12.558</td>
    <td>8.056</td>
    <td>8.856</td>
    <td>13.029</td>
    <td>15.361</td>
  </tr>
  <tr>
    <td>Kodak18gray.bmp</td>
    <td>394296</td>
    <td>10.709</td>
    <td>8.858</td>
    <td>10.11</td>
    <td>10.597</td>
    <td>12.161</td>
    <td>8.529</td>
    <td>9.473</td>
    <td>10.608</td>
    <td>11.641</td>
  </tr>
  <tr>
    <td>Kodak21.bmp</td>
    <td>1179704</td>
    <td>33.775</td>
    <td>10.741</td>
    <td>11.464</td>
    <td>13.93</td>
    <td>15.998</td>
    <td>10.337</td>
    <td>10.706</td>
    <td>18.817</td>
    <td>20.759</td>
  </tr>
  <tr>
    <td>Kodak21gray.bmp</td>
    <td>394296</td>
    <td>12.705</td>
    <td>10.909</td>
    <td>11.728</td>
    <td>12.747</td>
    <td>13.828</td>
    <td>10.513</td>
    <td>10.972</td>
    <td>13.054</td>
    <td>13.393</td>
  </tr>
  <tr>
    <td>Kodak22.bmp</td>
    <td>1179704</td>
    <td>33.94</td>
    <td>10.621</td>
    <td>12.252</td>
    <td>13.319</td>
    <td>16.184</td>
    <td>10.133</td>
    <td>11.338</td>
    <td>15.513</td>
    <td>18.437</td>
  </tr>
  <tr>
    <td>Kodak22gray.bmp</td>
    <td>394296</td>
    <td>12.866</td>
    <td>10.763</td>
    <td>12.441</td>
    <td>12.922</td>
    <td>15.111</td>
    <td>10.296</td>
    <td>11.547</td>
    <td>12.724</td>
    <td>14.189</td>
  </tr>
</table>