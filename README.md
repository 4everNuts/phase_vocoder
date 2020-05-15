# Phase Vocoder
This is an implementation of phase vocoder from http://www.guitarpitchshifter.com/algorithm.html
Phase vocoder allows sound pitch shifting

### Usage:
```python phase_vocoder.py INPUT_PATH OUTPUT_PATH SEMITONES_DIFF [--window=window_size]```  
Example:  
```python phase_vocoder.py data/terminator2.wav data/terminator_out.wav 1.2 --window=1024```