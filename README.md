# Fog / haze removal

Three implementations of fog/haze removal algorithms:

[1] 	FAST SINGLE IMAGE FOG REMOVAL USING EDGE-PRESERVING SMOOTHING; Jing Yu, Qingmin Liao   
[2]	Single Image Fog Removal Using Bilateral Filter; Abhishek Kumar Tripathi, Sudipta Mukhopadhyay   
[3] 	Atmospheric scattering-based multiple images fog removal; Zhang Tao, Shao Changyan ,Wang Xinnian    

### Prerequisites

```
[1]
* Octave with library 'image' installed. Should be able to run scripts from console.
* Python 3 or superior

[2],[3]
* Python 3 or superior
```

### Installing

No installation needed.

## Running 

```
[1]: run python3 fog_2011_edge_preserving/fog2011main.py
[2]: run python3 fog_2012_bilateral_filter/main.py
[3]: run python3 fog_2011_multyple_images/main.py
```

## Example

Results comparison with [1]* and [2]*   
![Example](/images/results.png)

## Authors

* **Alex Oarga** - *Github* - [alexOarga](https://github.com/alexOarga)

## Acknowledgments

* *fog_2011_edge_preserving / wlsfilter.m*  was obtained from:
 http://www.cs.huji.ac.il/~danix/epd/wlsFilter.m

