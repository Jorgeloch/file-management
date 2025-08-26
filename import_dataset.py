from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="LMUK-RADONC-PHYS-RES/TrackRAD2025",
    repo_type="dataset",
    local_dir="./data/"
)

