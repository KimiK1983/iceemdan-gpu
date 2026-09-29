# Colominas paper: Figures 1–15

The reference is Colominas et al. (2014), DOI [10.1016/j.bspc.2014.06.009](https://doi.org/10.1016/j.bspc.2014.06.009). The [authors' MATLAB code](https://zenodo.org/records/5580793) does not supply every random draw or plotted input vector. The [500 matched rows](../evidence/colominas_500_pairs.jsonl) compare this GPU implementation with the public Python CPU v2.0.0 using identical explicit noise. They are not an exact MATLAB reproduction.

| Figure | GPU status | Evidence and limit |
|---:|---|---|
| 1 | Method approximation | The implementation follows ICEEMDAN recurrence; the paper diagram is not copied. |
| 2 | Not reproduced | No GPU multi-method comparison with the authors' MATLAB implementations. |
| 3 | Partial, synthetic | Five ensemble sizes and 100 seeds each cover the ICEEMDAN arm only; the full multi-method panel is unavailable. |
| 4 | Partial, synthetic | First-component and first-residue RRSE are recorded per GPU/CPU pair, using known synthetic parts. |
| 5 | Partial, synthetic | Reconstruction RRSE is recorded per pair; completeness alone does not establish component identity. |
| 6 | Unresolved | The second synthetic signal's printed phase `-acos(13)` is not real-valued; its intended generator is needed. |
| 7 | Unresolved | The stated 1000-sample range conflicts with a plot extending to 2000; the intended vector is needed. |
| 8 | Candidate, GPU analyzed | [Keele](https://zenodo.org/records/3921794) `f1nw0000/laryngograph.wav`, 17.275–17.455 s, is a visual candidate. The [GPU metrics](../evidence/public_data_20260928/keele/metrics.json) record the local polarity inversion and archive hash. |
| 9 | Candidate, GPU analyzed | The Fig. 8 window produced 10 components and residue with 100 trials; reconstruction error was 1.82e-12 ADC. No paper seed or source array is available to verify equality. |
| 10 | Candidate, GPU analyzed | The same Keele record, 25.925–26.025 s, is a second candidate in the [GPU metrics](../evidence/public_data_20260928/keele/metrics.json). |
| 11 | Candidate, GPU analyzed | The Fig. 10 window produced 8 components and residue with 100 trials; reconstruction error was 1.82e-12 ADC. No samplewise paper equality is claimed. |
| 12 | Candidate, GPU analyzed | [CUDB v1.0.0](https://physionet.org/content/cudb/1.0.0/) `cu01`, 206–222 s, has a VFON annotation at 214.184 s. The paper does not identify its ECG record; see [GPU metrics](../evidence/public_data_20260928/cudb/metrics.json). |
| 13 | Candidate, GPU analyzed | The CUDB window produced 10 components and residue with 100 trials; reconstruction error was 2.27e-13 ADC. Source identity and paper component equality remain unverified. |
| 14 | Unresolved | The intracranial EEG patient record and 75 s sample vector are unidentified. |
| 15 | Unresolved | The same unidentified patient vector is required; no substitute patient waveform is included. |

The [Keele metrics](../evidence/public_data_20260928/keele/metrics.json) and [CUDB metrics](../evidence/public_data_20260928/cudb/metrics.json) contain source URLs, archive and record hashes, windows, parameters, versions, code hashes, component counts, stop reasons and reconstruction errors only. Both analyses ran with 100 trials, `epsilon=0.2`, seed 0, `batch_emd=True`, `graph_control=False`. Downloads and regenerated output stay under ignored `data/` and `outputs/`. No biomedical media or patient arrays are committed. Dataset licenses are separate from the code license. The [matched synthetic protocol](COLOMINAS_500_RESULTS.md) is the public CPU/GPU parity evidence.
