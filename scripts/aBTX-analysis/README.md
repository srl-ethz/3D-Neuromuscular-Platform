# AChR cluster analysis

Scripts that produce the AChR cluster plots.

| Script | Plot | Comparison |
|---|---|---|
| `achr_clusters_mn.py` | Number of AChR clusters | −MN vs +MN |
| `achr_innervation_flat_vs_grooves.py` | Number of AChR clusters, number of innervated clusters | Flat vs Grooves |

Each script reads Nikon `.nd2` z-stacks arranged as `<input>/<condition>/<replicate>/*.nd2`
(one stack per field of view). It writes one CSV with one row per biological replicate,
and the plots made from that CSV.

```bash
pip install numpy pandas scipy scikit-image matplotlib nd2
python achr_clusters_mn.py --input <input> --out <out>
python achr_innervation_flat_vs_grooves.py --input <input> --out <out>
```

To re-draw the plots from the CSV alone, without the images:

```bash
python achr_clusters_mn.py --csv <out>/achr_clusters_mn.csv --out <out>
python achr_innervation_flat_vs_grooves.py --csv <out>/achr_innervation_flat_vs_grooves.csv --out <out>
```

Parameters are listed at the top of each script.
