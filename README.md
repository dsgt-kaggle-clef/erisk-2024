# erisk-2024

This is the repository for DS@GT, eRisk challenge 2024.

## quickstart

Install the package for local development:

```bash
pip install -e .
```

Files are available at the following gcs location:

```bash
gcloud storage ls gs://dsgt-clef-erisk-2024
```

## running luigi tasks

Run the workflows using module syntax on one of the VMs.

```bash
python -m erisk.workflows.baseline
```
