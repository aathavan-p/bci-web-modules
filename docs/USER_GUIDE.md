# User guide

This guide is for students who use the modules. No programming is needed.

Start the app with `streamlit run app.py` and choose a module in the sidebar. Every module has the
same parts:

1. **Data source** (sidebar): the sample, PhysioNet, or your own file.
2. **Parameters** (sidebar): every change updates all plots right away.
3. **Plots**: they are linked, so clicking in one changes the others.
4. **💡 Explain**: what you are seeing, with the numbers from your current settings, plus questions to
   check your understanding.
5. **⬇️ Export**: a Python script that reproduces your view, the parameters as JSON, and the data.

---

## Module 1 · Data loading

**Goal:** understand what an EEG recording contains before you process it.

### What is on screen

| Element | What it shows |
|---|---|
| Metrics row | Channels **C**, samples **T**, sampling rate, duration, number of annotations |
| Raw traces | The selected channels in the current window, in µV. Shaded areas are annotations. The black bar is the amplitude scale. The orange trace is the focus channel. |
| Head map | Each electrode seen from above (nose up), coloured by RMS, peak-to-peak or variance z-score in the window. **Click a sensor** to add it to the traces and make it the focus. |
| Spectrum | Power spectrum of the focus channel in the window. The alpha/mu and beta bands are shaded. |
| Navigator | Every annotation in the whole recording. The orange box is your window. **Click an event** to jump to it. |
| File anatomy tab | The dimensions table, annotation counts, the events array MNE builds, the channel table and the full `raw.info` |

### Try these

1. **Find the blinks.** On the sample, show Fp1 and Fp2. Large, slow bumps appear every few seconds.
   Switch the head map to *variance z-score*: Fp1 and Fp2 are flagged as outliers.
2. **See the mains noise.** Focus on any channel. The spectrum has a sharp peak at 50 Hz.
3. **Make an alias.** Set *Resample to* 60 Hz and untick *Anti-alias filter*. The red spectrum now has a
   peak at **10 Hz**, in the alpha band, though no 10 Hz source was added. Tick the filter again and
   it disappears. The Explain panel gives the exact alias frequency for any rate you choose.
4. **Compare real EEG.** Switch to PhysioNet, subject 1, run 4. Note the 64 channels, the 160 Hz rate
   and the T0/T1/T2 events, renamed to rest / left_hand / right_hand.
5. **Load your own file.** Upload a `.fif`, `.edf` or `.csv` from the lab session (Component 1).

---

## Module 7 · Feature extraction

**Goal:** turn trials into a small feature matrix **X** (trials × features) and judge which features
separate the classes.

### What is on screen

| Element | What it shows |
|---|---|
| Metrics row | Number of trials, trial size (C × T), features per trial, cross-validated accuracy, and the accuracy needed to beat chance at p < .05 |
| Spectrum by class | Mean ± SEM spectrum at the focus channel for each class. The orange band is the feature band. |
| Head map | *Band power / Hjorth:* Fisher score per channel (bright means discriminative). *CSP:* the spatial pattern of the chosen component. **Click a sensor** to change the focus channel and put its feature on the scatter x-axis. |
| Feature scatter | One dot per trial in a 2-D feature space. **Click a dot** to overlay that trial's spectrum. |
| Fisher bars | The 20 most discriminative features. **Click a bar** to plot that feature. |
| Feature matrix | The table the classifier would receive |
| Band sweep | CV accuracy for a 4 Hz band moved from 4 to 42 Hz. Shows where the information is. |

### Try these (sample data, left vs right hand)

1. **Band power, 8–13 Hz.** The head map lights up at C3 and C4. In the spectrum at C3, *right_hand* is
   lower (desynchronisation on the side opposite the hand). Click C4: now *left_hand* is lower.
2. **Wrong band.** Choose *low gamma* (30–45 Hz). Accuracy falls to around chance, and the head map
   goes dark.
3. **Wrong time.** Set the epoch window to −1.5 … 0 s, before the cue. Accuracy is at chance, as it
   should be. If it were not, something would be leaking.
4. **CSP.** Switch to CSP. Four features now replace 26, and the patterns of CSP1 and CSP2 sit over C4
   and C3.
5. **Frequency resolution.** In band power, set the Welch segment to 0.25 s. The spectrum becomes
   blocky (4 Hz bins), and a narrow band may contain no bins at all.
6. **Run the band sweep** for band power, then for CSP, and compare.

### How to read the accuracy

The accuracy is a **sanity check of the features**, not a full classifier evaluation (Module 8 covers
that). It uses stratified k-fold cross-validation with a shrinkage LDA. The feature extractor (CSP
included) and the scaler are **re-fitted inside every training fold**, so test trials never influence
the features. The chance threshold comes from a binomial test: with 40 trials you need about 62 %,
not 50 %.

---

## Exported scripts

Download the `.py` file from the Export panel. Place it in the repository root and run
`python <file>.py`. It recreates the data source (synthetic, PhysioNet or your file, which must be
in the same folder) and repeats the same steps with the same parameters.

## Troubleshooting

| Problem | Fix |
|---|---|
| PhysioNet download fails | Check your internet connection or proxy. The sample works offline. |
| "Need at least two classes" | Your file has no labelled events. Use the sample or PhysioNet run 4/8/12, or add annotations to your file. |
| No head map | The file has no electrode positions. Use standard 10-20 channel names (C3, Cz …) so they can be matched. |
| `.vhdr` or `.set` with a separate `.fdt` will not upload | Convert to `.fif` first: `mne.io.read_raw(...).save("x_raw.fif")`. |
