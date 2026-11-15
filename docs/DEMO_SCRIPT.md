# 10-minute demo script (every member speaks)

Written for a group of **3**. With 4 members, split segment 3 into two parts (3a: band power and
the wrong-window checks, 3b: CSP and the band sweep).
Rehearse with a timer. Start the app **before** the demo, and open PhysioNet run 4 once beforehand
so the download is already cached.

**Setup:** `python -m streamlit run app.py`, browser zoom at 90 %, sidebar visible.

---

## 1 · Why and how · 0:00–2:00 · Member A

*Screen: app home, Data loading selected.*

- "A future student should be able to change a parameter and immediately see its effect on EEG.
  We built two steps of the pipeline: **loading** and **feature extraction**."
- Show the module switcher in the sidebar. "Each module is a folder that exposes one `render()`
  function, so the two merge into the course app without changes." (5 seconds on the folder tree in
  your editor or on GitHub.)
- Three data sources: the **synthetic sample**, which has a known ground truth, **real PhysioNet
  EEG**, and **your own file**.

## 2 · Data loading · 2:00–5:00 · Member B

*Sample data.*

1. **Metrics row** (15 s): "26 channels × 62,960 samples at 160 Hz. This C × T matrix is what every
   later step works on."
2. **Traces + navigator** (30 s): click a `left_hand` event in the navigator, and the window jumps there
   with the trial shaded.
3. **Blinks** (30 s): point at the large bumps on Fp1. Switch the head map to *variance z-score*:
   Fp1 and Fp2 are flagged. "Blinks: large, slow, frontal."
4. **Linked head map** (20 s): click C4 on the head map. It is added to the traces, and the spectrum
   switches to C4.
5. **Aliasing, the highlight** (60 s): set *Resample to* 60 Hz and **untick anti-alias**. A red peak
   appears at **10 Hz**. "50 Hz mains noise has folded into the alpha band, so it would look like a brain
   rhythm." Tick the filter again and it is gone. Open **Explain**: the text gives the exact alias
   frequency.
6. **Real data** (25 s): switch to PhysioNet subject 1, run 4. "64 channels; T0/T1/T2 renamed to
   rest / left_hand / right_hand." Open the *File anatomy* tab and show the events array.

## 3 · Feature extraction · 5:00–8:30 · Member C

*Sample data, left vs right hand, band power, alpha/mu.*

1. **The idea** (20 s): "Each trial is 26 × 481 numbers, and we reduce it to 26 features."
   Point at the metrics row.
2. **Head map + spectrum** (45 s): the bright spots are C3 and C4. Click C3: *right_hand* has lower
   mu power there. Click C4: now *left_hand* does. "Contralateral desynchronisation, exactly as in
   the lectures."
3. **Scatter** (25 s): the two clusters. Click one trial, and its spectrum is overlaid as the dotted line.
4. **Accuracy and chance** (30 s): "80 % with cross-validation. With 40 trials you need **62 %**, not
   50 %, to beat chance at p < .05."
5. **Two ways to fool yourself** (40 s): choose *low gamma*, and accuracy falls to about chance. Reset the
   band, then drag the epoch window to −1.5…0 s (before the cue): chance again. "If either of these
   had looked good, we would have a bug."
6. **CSP** (30 s): switch the method. "Four features instead of 26. The patterns sit over C3 and C4. CSP
   uses the labels, so we re-fit it inside every fold, and a test with shuffled labels proves it."
7. **Band sweep** (20 s): run it. The accuracy curve peaks in the mu/beta range.

## 4 · Quality and wrap-up · 8:30–10:00 · Member A (or D)

- **Export** (20 s): open the Export panel. "This script reproduces exactly this view, and our tests
  run every exported script."
- **Tests** (20 s): show `pytest -q` → 24 passed, and the green CI badge or Actions run on GitHub.
- **Docs** (10 s): README, user guide and developer guide.
- **Limitations** (20 s): the synthetic data is simpler than real EEG, the accuracy is a feature
  check rather than a full evaluation (that is Module 8), and multi-file formats (.vhdr) cannot be
  uploaded.
- "Questions?"

---

## Likely viva questions (prepare one owner each)

| Question | Short answer |
|---|---|
| Why does naive resampling alias? | Frequencies above the new Nyquist (fs/2) cannot be represented and fold to \|f − k·fs\|. A low-pass filter must remove them first. |
| Why log band power? | Power is positive and skewed. Its log is closer to Gaussian, which LDA assumes. |
| CSP filter vs pattern? | The filter extracts the source. The pattern shows how the source projects to the scalp. Only the pattern is interpretable as a map. |
| Where could leakage happen in your pipeline? | Fitting CSP, the scaler or band choice on all trials. We fit inside folds and test with shuffled labels. Choosing the band by looking at the accuracy is still a form of selection bias. |
| How is the chance threshold computed? | A binomial inverse CDF at α = .05 for n trials and 1/k chance (Combrisson & Jerbi, 2015). |
| How is the synthetic data made? | 1/f background plus narrowband sources projected through Gaussian spatial weights, a trial-wise ERD envelope on C3/C4, blinks and 50 Hz noise, all from a fixed seed. |
| Frequency resolution of Welch? | 1 / segment length. 1 s gives 1 Hz bins. |
