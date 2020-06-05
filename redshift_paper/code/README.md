# How to Run Code for Redshift Paper 


## Package Requirements
```
colossus==1.2.15
matplotlib==3.1.0  (2.0.2<version<3.2.1)
numpy==1.16.4 (1.13.3<version<1.18.3)
pandas==0.25.1  (0.20.1<version<1.0.3)
seaborn==0.9.0 (>0.7.1)
```


## Code

### Plotting Scripts:

```
./pair_plot.py
./phase_space.py
./selection_effects.py
./sizehist_new.py
./zplot_final.py
./red_sequence.py
./create_gif.py
```

### Helper Functions
These scripts are called in other scripts:
`read_data.py` and `conversions.py`



###Useful Calculators:

#### Splashback Radius

```
./splashback.py
```

#### Angular Separation Calculator
Requires (ra1,dec1) and (ra2,dec2) input at run-time.

```
./ra_dec_offset.py
```

#### Angular Separation <--> Physical Separtion Calculator

To convert 300 kpc at cz=7000 km/s to angular separtion (in arcsec).

```
python convert_kpc_arcsec.py 7000 300 0
```

To convert 10 arcsec at cz=7000 km/s to physical separation (in kpc).

```
python convert_kpc_arcsec.py 7000 0 10
```

#### Feature Correlations

```
./correlation.sh
```

or 

```
./correlation.py kadowaki2019.tsv       \
                 --udgs                 \
                 --table all            \
                 --environment all      \
                 --verbose              \
> min_comparisons.txt
```

---

### TODO (Paper):
1. 
2. 

### TODO (Code):
1. Update `convert_kpc_arcsec.py` and `ra_dec_offset.py` to use Python3 & `argparse`.
2. Separate `pair_plot.py` into separate scripts. 
3. Fix bugs:

	(3a) density=True (need to set local_env=False) to get correct colors in density pair plots. 	
	 
	(3b) Occurs when UDG has: LocalEnv=Unconstrained. Need to apply a velocity cut.
	
	```
	~~~~~~LOCAL~~~~~~
Traceback (most recent call last):
  File "./pair_plot.py", line 521, in <module>
    hack_color_fix=False)
  File "./pair_plot.py", line 386, in main
    hue_kws={"marker":markers})
  File "/Users/jkadowaki/anaconda3/lib/python3.6/site-packages/seaborn/axisgrid.py", line 1249, in __init__
    self.palette = self._get_palette(data, hue, hue_order, palette)
  File "/Users/jkadowaki/anaconda3/lib/python3.6/site-packages/seaborn/axisgrid.py", line 158, in _get_palette
    color_names = [palette[h] for h in hue_names]
  File "/Users/jkadowaki/anaconda3/lib/python3.6/site-packages/seaborn/axisgrid.py", line 158, in <listcomp>
    color_names = [palette[h] for h in hue_names]
KeyError: 'Unconstrained'
	```

    (3c) Deprecation Warning
    
    ```
    ./sizehist_new.py:80: MatplotlibDeprecationWarning: 
The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
  facecolor='darkseagreen', alpha=0.45)
./sizehist_new.py:82: MatplotlibDeprecationWarning: 
The 'normed' kwarg was deprecated in Matplotlib 2.1 and will be removed in 3.1. Use 'density' instead.
  n, bins, patches = plt.hist(t['re'],10, normed=1, facecolor='r', alpha=0.3)
    ```