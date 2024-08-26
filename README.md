# density_regression
## <a name="demo"></a> Quick Demo
Run [this Google Colab](https://colab.research.google.com/drive/1p5gK-rOI4XYgg2zTVtbh-5Ky06PGlA09?usp=sharing).

or

notebook in `density_regression.ipynyb`

or 

python file (full comparision, install prerequisite packages first to import library):
```sh
python demo/run_cubic.py
python demo/density_regression.py
```

## <a name="prepare"></a> To prepare:
### <a name="library">Library</a>
Install prerequisite packages:
```sh
pip install -r requirements.txt
```

### <a name="dataset">Dataset</a>
Download dataset:
```sh
bash depth_estimation/download_data.sh
```

## <a name="experiments"></a> To run experiments:
### <a name="Time series weather forecasting">Time series weather forecasting</a>
```sh
python <method_file> --exp_idx=<idx>
```
where the parameters are the following:
- `<method_file>`: file stored the code of method. E.g., `<method_file> = time_series/density_regression.py`
- `<idx>`: index of experiment. E.g., `<idx> = 1`

### <a name="Benchmark UCI">Benchmark UCI</a>
```sh
python uci/main.py --datasets=<dataset_name> 
```
where the parameters are the following:
- `<dataset_name>`: name of the sub-dataset in UCI. E.g., `<dataset_name> = "wine"`

### <a name="Monocular depth estimation">Monocular depth estimation</a>
```sh
python depth_estimation/main.py --model=<method_name> 
```
where the parameters are the following:
- `<method_name>`: name of method. E.g., `<method_name> = "densityregressor"`

## References
Based on code of:
> [Time series forecasting](https://www.tensorflow.org/tutorials/structured_data/time_series#baseline)\
> TensorFlow.

> [Evidential Deep Learning](https://github.com/aamini/evidential-deep-learning)\
> Amini, Alexander and Schwarting, Wilko and Soleimany, Ava and Rus, Daniela.\
> _arXiv:1910.02600_.

> [Methods for comparing uncertainty quantifications for material property predictions](https://github.com/ulissigroup/uncertainty_benchmarking)\
> Kevin Tran, Willie Neiswanger, Junwoong Yoon, Eric Xing, Zachary W. Ulissi.\
> _arXiv:1912.10066_.


## License
This source code is released under the Apache-2.0 license, included [here](LICENSE).
