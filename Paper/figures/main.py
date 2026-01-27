from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import seaborn as sns
from h5py import Dataset
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from scipy.signal import spectrogram

DATA_DIR = Path("./data")
OUT_DIR = Path("./out")
SAMPLE_RATE = 48_000


def extract_audio(file: Path, video_id: str, start_sample: int, stop_sample: int):
    f = h5py.File(file)
    ds = f[video_id]
    if not isinstance(ds, Dataset):
        raise ValueError(f"Wrong dataset type: {type(ds)}")
    return ds[start_sample:stop_sample]


def two_example_spectograms():
    audio_p1 = extract_audio(
        DATA_DIR / "P01_audio.hdf5",
        "P01-20240202-110250",
        15_360_000,
        15_648_000,
    )

    audio_p2 = extract_audio(
        DATA_DIR / "P02_audio.hdf5",
        "P02-20240209-184316",
        2_400_000,
        2_688_000,
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 4), sharey=True)

    for idx, audio in enumerate((audio_p1, audio_p2)):
        f, t, S = spectrogram(audio, SAMPLE_RATE, nperseg=512, noverlap=384)

        ax: Axes = axes[idx]
        ax.pcolormesh(t, f, 10 * np.log10(S + 1e-10), cmap="magma")
        ax.set_xlabel("Time (s)")
        ax.set_title(f"P0{idx + 1}")

    axes[0].set_ylabel("Frequency (Hz)")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "hd_epic_ex_specto.png", dpi=600, bbox_inches="tight")


def repro_boxplots():
    df = pd.read_csv("./data/repro.csv")

    df = df.rename(
        columns={"Paper Score": "Original", "Reproduction Score": "Reproduction"}
    )

    diff = (df["Original"] - df["Reproduction"]).abs()

    print(f"{diff.min()=}, {diff.max()=}, {diff.mean()=}")

    df_melt = df.melt(
        id_vars="dataset",
        value_vars=["Original", "Reproduction"],
        var_name="metric",
        value_name="value",
    )

    sns.boxplot(x="dataset", y="value", hue="metric", data=df_melt)
    plt.title("F1-Score (%) by Dataset (Averaged over LOO Evaluation)")
    plt.legend(title="", loc="lower left")
    plt.xlabel("Dataset")
    plt.ylabel("F1-Score (%)")

    plt.tight_layout()
    plt.savefig(OUT_DIR / "repro_boxplots.png", dpi=600)


def main():
    # annotations = pd.read_csv("./data/HD_EPIC_Sounds.csv")
    # print(annotations["class"].unique())
    # annotations = annotations[annotations["video_id"] == "P01-20240202-110250"]
    # # annotations = annotations[annotations["video_id"] == "P02-20240209-184316"]
    # annotations = annotations[annotations["class"] == "water"]
    # print(annotations)

    DATA_DIR.mkdir(exist_ok=True, parents=True)
    OUT_DIR.mkdir(exist_ok=True, parents=True)

    two_example_spectograms()
    repro_boxplots()


if __name__ == "__main__":
    main()
