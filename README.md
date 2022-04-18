# HDR CHIPQA: No-Reference Video Quality Assessment for HDR using Space-Time Chips

This repository contains code for HDR ChipQA and ChipQA.

## Requirements

Create a conda environment from the specification file  hdr_chipqa_spec-file.txt using
```

conda create --name hdr_chipqa --file conda_spec-file.txt

```
Activate the environment. There are some packages only pip can install, so also do 

```

pip install -r pip_requirements.txt

```

You should now be good to go!

## Feature extraction

To extract features, run (for eg.)
```

python3 hdr_chipqa.py --input_file I.yuv --results_file O.z --width 3840 --height 2160 --bit_depth 10 --color_space BT2020

```

## Training with LIVE-HDR VQA database

Run 
```

python3 hdrchipqa_svr.py --score_file score.csv --feature_folder ./folder --train_and_test

```
to evaluate. Other options can be seen with the -h option.


## Testing on a new database

1. Run hdr_chipqa.py with the path to the folder of videos and the output directory.
2. Run `testing.py` with a path to the input feature file(s).
