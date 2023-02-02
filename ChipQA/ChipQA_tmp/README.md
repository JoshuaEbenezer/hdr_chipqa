# CHIPQA: No-Reference Video Quality Assessment using Space-Time Chips

This repository contains code for ChipQA and ChipQA-0.

## Requirements

See requirements.txt

## Feature extraction

To extract features, run
```

python3 chipqa.py path/to/input_folder path/to/output_folder

```
`input_folder` must contain the videos from which chipqa features will be extracted.
`output_folder` will contain the features that are written out.

## Training with LIVE-APV Livestream VQA database

After extracting the features, run 
```

python3 zip_feats_and_scores.py input_folder csv_file output_file 

```
Here `input_folder` must contain the features generated from the previous step, `csv_file` must be a csv file with LIVE-APV database names and scores, and `output_file` will be where the combined features and scores will be stored.

After this, run 
```

python3 svr.py input_file output_folder

```
to evaluate. `input_file` must be the path where the zipped features and scores are saved. The predictions and the ground truth MOS for each of the runs will be stored in `output_folder`.

`all_srocc.m` can then be used to find the SROCC and LCC.

## Testing on a new database

1. Run chipqa.py with the path to the folder of videos and the output directory.
2. Run `testing.py` with a path to the input feature file(s).
