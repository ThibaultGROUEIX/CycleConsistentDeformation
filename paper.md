# Reproduce the paper results



* Launch all the trainings with `scripts/launch.py --mode training` (remove the trained models in `/log` if you downloaded them for the demo) OR call `./log/download_pretrained_models.sh`.
* Launch all the inferences in `scripts/launch.py --mode inference

This populates a global Pandas dataframe with the results called `results.csv`.

* Make the tables : `python scripts/make_papertables_from_results.py` and check them in the `html/` folder.

* Make the curves : Run the notebook : The values are hard-coded from the results of my own experiments.

* Make the figures : check the generated mesh in the `figures/` folder 

