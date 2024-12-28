# pytorch-mnist

This repository serves as a practice project, encompassing a hand-written digit recognition model based on the MNIST dataset, implemented using PyTorch.

## File Structure

| Path               | Description                                |
|:-------------------|:-------------------------------------------|
| `/commands/...`    | Command scripts.                           |
| `/data/...`        | Data loader script and downloaded dataset. |
| `/models/...`      | Model definitions.                         |
| `/notebooks/...`   | Analysis notebook templates.               |
| `/src/...`         | Loop functions.                            |
| `/main.py`         | Entry point script.                        |
| `/environment.yml` | Conda environment definition.              |

## Usage

1. (optional) Edit training config in `main.py`;
2. Create [conda environment](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) from `environment.yml`:

    ```bash
    conda env create -f environment.yml
    ```

3. Execute `main.py` and wait for it to finish:

    ```bash
    python main.py train
    ```

4. Open `/path/to/log/log.ipynb`(the actual path is available in the output of last command), run it, and view the results.

Execute `python main.py --help` to get an overview of available commands.
