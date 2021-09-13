# FastQA

This repository contains efficient python implementations of some QA metrics. Over time, I will try to grow the list of available metrics.
These metrics are "efficient" in the sense that they use rectangular windows for all averaging operations, thereby leveraging integral images (also known as summed area tables) to compute averages efficiently.

Even though this code is written in numpy, it can easily be modified to use with deep learning libraries like PyTorch and Tensorflow.
