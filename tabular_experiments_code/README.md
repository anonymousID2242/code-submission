## Tabular code experiments
## Run Instructions
Run the code for the examples using -
```
In tabular_experiments_code :
To run MAIRE Code :

cd src/
python -m examples.adult

To compare methods :

cd compare_lime_anchors/
python2 make_graph_plots.py

```

The results will be produced in the compare_lime_anchors/graphs folder.\
To retrain a classifier that is explained, comments model.load_weights and rerun using above commands.\







