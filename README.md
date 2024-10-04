# pytorch-mnist

This repository serves as a practice project, encompassing a hand-written digit recognition model based on the MNIST dataset, implemented using PyTorch.

## File Structure

| Path                | Description                                                    |
|:--------------------|:---------------------------------------------------------------|
| `/data/`            | This folder contains a loader script and downloaded dataset.   |
| `/model/`           | Here defines the PyTorch model.                                |
| `/notebook/`        | Analysis notebooks are located here.                           |
| `/src/`             | Here defines the loop functions: `train_loop` and `eval_loop`. |
| `/main.py`          | The training script.                                           |
| `/requirements.txt` | Python dependencies are listed in this file.                   |

## Usage

1. (optional) Edit training config in `main.py`;
2. Create and activate [python virtual environment](https://docs.python.org/3/library/venv.html);
3. Install dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Execute `main.py` and wait for it to finish:

    ```bash
    python main.py
    ```

5. Open `/notebook/log.ipynb` and update `log_dir` which should be
    a `pathlib.Path` object pointing to the most recent log folder;
6. Re-run the notebook to view the results.
