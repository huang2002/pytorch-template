# pytorch-template

This is a personal template repository for PyTorch projects. A hand-written digit recognition model is included as example.

## File Structure

| Path               | Description                                |
|:-------------------|:-------------------------------------------|
| `/commands/...`    | Command scripts.                           |
| `/data/...`        | Data loader script and downloaded dataset. |
| `/models/...`      | Model definitions.                         |
| `/notebooks/...`   | Analysis notebook templates.               |
| `/main.py`         | Entry point script.                        |
| `/environment.yml` | Conda environment definition.              |

## Usage

1. Open `environment.yml` and edit the environment name in the first line.
2. Create [conda environment](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) from `environment.yml` and activate it:

    ```bash
    conda env create -f environment.yml
    conda activate <YOUR_ENVIRONMENT_NAME>
    ```

3. Execute `main.py` and wait for it to finish:

    ```bash
    python main.py train
    ```

4. Open `/path/to/log/statistics.ipynb`(the actual path is available in the output of last command), run it, and view the results.

Execute `python main.py --help` to get an overview of available commands.
