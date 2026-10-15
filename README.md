# NM-swimmers-analysis

![Python 3.9](https://img.shields.io/badge/python-3.9-blue)
![License: MIT](https://img.shields.io/badge/license-MIT-green)

This repository contains the code used for data analysis and figure generation of the main figures of *A 3D Human Neuromuscular Model with Biomimetic Architecture Encoded by Compliant Patterned Hydrogel Substrates*.

## Repository structure

`scripts/` contains the analysis and plotting scripts, with one subfolder per analysis. Instructions are detailed at the top of each script.

- `aBTX-analysis` — AChR clusters and innervated clusters
- `neuron-analysis` — axon alignment, area and length
- `muscle-analysis` — myotube fusion index, fiber density and alignment
- `coalignment-analysis` — axon–muscle overlap, co-aligned vs orthogonal
- `calcium-analysis` — active Ca²⁺ area before and after glutamate
- `glutamate-response` — muscle contraction after glutamate and curare
- `injury-contraction` — bending angle after injury

## Installation and requirements

The code was run with Python 3.9. Dependencies are listed in `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Usage

Run the scripts from the repository root. The usage of each script is given at the top of the file.


## License

MIT License. See [LICENSE](LICENSE).
