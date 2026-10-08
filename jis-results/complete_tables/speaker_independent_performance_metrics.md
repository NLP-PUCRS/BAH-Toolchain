*Table: Performance metrics of different feature extractors using SVC on the Verbo dataset in a speaker-independent setting (averaged across seeds 42, 97, 123, 2025). The speakers on the test set for each seed were: 42 ['f6', 'm1'], 97 ['f2', 'm4'], 123 ['f1', 'm3'], 2025 ['f5', 'm1'].*

| Model | Balanced Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| HuBERT | $0.3380 \pm 0.0866$ | $0.3512 \pm 0.0841$ | $0.3380 \pm 0.0866$ | $0.3308 \pm 0.0881$ |
| Whisper | $0.4490 \pm 0.0896$ | $0.4972 \pm 0.0873$ | $0.4490 \pm 0.0896$ | $0.4460 \pm 0.0928$ |
| wav2vec2 | $0.3380 \pm 0.0866$ | $0.3512 \pm 0.0841$ | $0.3380 \pm 0.0866$ | $0.3308 \pm 0.0881$ |
| WavLM | $0.2436 \pm 0.0307$ | $0.2446 \pm 0.0681$ | $0.2436 \pm 0.0307$ | $0.2259 \pm 0.0391$ |
| FRILL | $0.3571 \pm 0.0515$ | $0.4271 \pm 0.0818$ | $0.3571 \pm 0.0515$ | $0.3495 \pm 0.0592$ |
| eGeMAPSv02_88 | $0.3010 \pm 0.0325$ | $0.3297 \pm 0.0617$ | $0.3010 \pm 0.0325$ | $0.2879 \pm 0.0388$ |
| pAA | $0.3112 \pm 0.0480$ | $0.3367 \pm 0.0267$ | $0.3112 \pm 0.0480$ | $0.3039 \pm 0.0443$ |
| VGGish | $0.3240 \pm 0.0465$ | $0.3448 \pm 0.0804$ | $0.3240 \pm 0.0465$ | $0.3134 \pm 0.0493$ |
| Trillsson5 | $0.6059 \pm 0.0722$ | $0.6571 \pm 0.0768$ | $0.6059 \pm 0.0722$ | $0.5964 \pm 0.0832$ |
| ComParE_2016_6k | $0.3380 \pm 0.1219$ | $0.4302 \pm 0.0874$ | $0.3380 \pm 0.1219$ | $0.3270 \pm 0.1224$ |
| TRILL | $0.3520 \pm 0.0279$ | $0.3646 \pm 0.0107$ | $0.3520 \pm 0.0279$ | $0.3261 \pm 0.0182$ |