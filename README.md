# Image Comparison

### Description

The following program can be used to calculate the mean squared difference of two images in python using GPU accelerated computing. It specifically operates on **Pillow** Image objects, and utilises the **pyopencl** module.

### Usage

The two images the program will read by default are *image0.png* and *image1.png* and there is currently no way to change this without editing the source code. 

**Windows:**

```bash
python msd.py
```

The program has not been tested on any other operating system other than Windows.

### Dependencies

The following dependencies were used to create and run the program.

```
pyopencl Version 2023.1.4
numpy Version 1.23.5
Pillow Version 9.4.0
platformdirs Version 3.11.0 (Required by pyopencl)
pytools Version 2023.1.1 (Required by platformdirs)
```

