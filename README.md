# watermarking_with_qr_decomposition

Color Image Blind Watermarking with QR Decomposition
Description
This project implements a blind watermarking algorithm for color images using QR decomposition. The algorithm embeds a color watermark into a color host image, ensuring robustness against various image processing attacks while maintaining high watermark invisibility.

Key Features:
Blind Extraction: Watermarks can be extracted without the need for the original image or watermark.
Robustness: Resistant to common attacks like image compression, filtering, noise addition, cropping, scaling, and more.
Efficiency: Utilizes QR decomposition, which is computationally efficient compared to other methods like SVD.
Algorithm Highlights
Watermark Embedding:

The color host image is divided into 4x4 non-overlapping pixel blocks.
Each block undergoes QR decomposition.
Watermark bits are embedded by quantifying specific elements of the R matrix.
Watermark Extraction:

The watermarked image is decomposed to extract watermark bits from the R matrix.
No requirement for the original image or watermark.
Advantages
Higher robustness and invisibility compared to traditional methods.
Supports embedding color watermarks instead of binary or grayscale.
Maintains a balance between computational efficiency and watermark quality.
Usage
Watermark Embedding

Input: Host image (color), watermark image (color), and security keys.
Output: Watermarked image.
Watermark Extraction

Input: Watermarked image and security keys.
Output: Extracted watermark.
Applications
Digital copyright protection.
Authenticating multimedia content.
Securing sensitive image data.

Dependencies
Python 3.x
NumPy
OpenCV
