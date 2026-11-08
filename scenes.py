from math import gcd
from pathlib import Path
from functools import lru_cache
import numpy as np
import soundfile as sf
import torch
from scipy.signal import resample_poly
from torch.utils.data import Dataset

from ambisonics import encode_foa
from config import SR, N_SAMPLES, TARGET_RMS, MIN_SEP_DEG, MAX_MOVE_DEG

#Mélange deux sources dans une scène ambisonique.
#az1, az2 : nombre (source fixe) ou tableau (T,) (source mobile).
def make_scene(s1, s2, az1, az2):
    return encode_foa(s1, az1) + encode_foa(s2, az2)

@lru_cache(maxsize=None)
def load_audio(path):
    x, sr = sf.read(str(path), dtype="float32")
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        g = gcd(SR, sr)
        #On sur-echantillonne a SR//g puis on sous echantillone a sr//g,
        # le pgcd (gcd) nous permet davoir des nombre plus petit et facilite le calcul
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


#Extrait n échantillons au hasard (complète par des zeros  (silence) si le fichier est trop court)."""
def random_clip(x, n, rng):
    if len(x) < n:
        x = np.pad(x, (0, n - len(x)))
    start = rng.integers(0, len(x) - n + 1)
    return x[start:start + n]


def normalize_rms(x, rms=TARGET_RMS):
    return x * (rms / (np.sqrt(np.mean(x ** 2)) + 1e-8))


def list_audio_files(folder):
    folder = Path(folder)
    files = sorted(list(folder.rglob("*.flac")) + list(folder.rglob("*.wav")))
    return files


def speaker_of(path):
    return Path(path).parent.parent.name


class SceneDataset(Dataset):
    """Génère des scènes à deux locuteurs à la volée.

    files           : liste de fichiers audio
    n_items         : nombre de scènes par époque
    seed            : None = aléatoire à chaque époque (entraînement) ;
                      un entier = scènes toujours identiques (validation / test)
    moving          : True = les locuteurs se déplacent pendant le clip
    max_move_deg    : déplacement maximal (en degrés) sur la durée du clip
    fixed_target_az : si défini, la voix cible est toujours à cet azimut (ex. 0.0 = devant)
    min_sep_deg     : écart angulaire minimal entre les deux locuteurs au départ
                      (valeur par défaut dans config.py ; une valeur négative
                      ne garantit aucun écart, voir le commentaire de MIN_SEP_DEG)
    """

    def __init__(self, files, n_items=1000, seed=None, moving=False,
                 max_move_deg=MAX_MOVE_DEG, fixed_target_az=None, min_sep_deg=MIN_SEP_DEG):
        self.files = list(files)
        self.n_items, self.seed = n_items, seed
        self.moving, self.max_move_deg = moving, max_move_deg
        self.fixed_target_az, self.min_sep_deg = fixed_target_az, min_sep_deg

    def __len__(self):
        return self.n_items
# Il est question de la fonction de faire varier ou non le deplacement de la source  vers l'avant ou l'arriere
    def _trajectory(self, az_start, rng):
        if not self.moving:
            return np.full(N_SAMPLES, az_start)
        delta = rng.uniform(-self.max_move_deg, self.max_move_deg)
        return az_start + np.linspace(0, delta, N_SAMPLES)

    def __getitem__(self, idx):
        rng = np.random.default_rng(None if self.seed is None else self.seed + idx)

        # 1) Deux fichiers de locuteurs différents
        f1 = self.files[rng.integers(len(self.files))]
        f2 = f1
        for _ in range(20):
            f2 = self.files[rng.integers(len(self.files))]
            if speaker_of(f2) != speaker_of(f1):
                break
        s1 = normalize_rms(random_clip(load_audio(f1), N_SAMPLES, rng))
        s2 = normalize_rms(random_clip(load_audio(f2), N_SAMPLES, rng))

        # 2) Directions (cible = source 1, interférent = source 2)
        az1 = self.fixed_target_az if self.fixed_target_az is not None else rng.uniform(-180, 180)
        az2 = az1 + rng.uniform(self.min_sep_deg, 360 - self.min_sep_deg)
        traj1, traj2 = self._trajectory(az1, rng), self._trajectory(az2, rng)

        # 3) Scène ambisonique
        foa = make_scene(s1, s2, traj1, traj2)
        return {
            "foa": torch.from_numpy(foa),                       # (4, T)
            "target": torch.from_numpy(s1),                     # (T,) voix à extraire
            "interferer": torch.from_numpy(s2),                 # (T,)
            "az": torch.from_numpy(traj1.astype(np.float32)),   # (T,) azimut de la cible
        }