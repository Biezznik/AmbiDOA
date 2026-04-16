import os
import json
import torch
from torch.utils.data import Dataset
import torchaudio
import numpy as np
from AmbiDOA import log_print
from datetime import datetime

class AmbiDOADataset(Dataset):
    """
    Dataset do zadania lokalizacji wielu źródeł (DOA) na bazie manifestu JSON:
    [
      {
        "audio_filepath": "...",
        "n_sources": 1,
        "regions": [...],
        "azimuths": [...],      # [max_sources] (float lub None)
        "elevations": [...],    # [max_sources] (float lub None)
        "mask": [...]           # [max_sources] (0/1)
      },
      ...
    ]
    """
    def __init__(self, manifest_path, audio_dir, sr=11025, window_len_ms=250, dbfs_threshold=-80.0, max_sources=12):
        with open(manifest_path, "r") as f:
            self.entries = json.load(f)
        self.audio_dir = audio_dir
        self.sr = sr
        self.window_len = int(window_len_ms * sr / 1000)
        self.dbfs_threshold = dbfs_threshold
        self.max_sources = max_sources
        

        self.windows = []
        for item in self.entries:
            audio_path = os.path.join(audio_dir, item['audio_filepath'])
            info = torchaudio.info(audio_path)
            n_samples = info.num_frames
            n_full_windows = n_samples // self.window_len

            # Upewnij się, że manifest ma stałą długość list (mask/az/el) = max_sources
            azimuths   = _pad_or_trim(item.get('azimuths', []), max_sources, None)
            elevations = _pad_or_trim(item.get('elevations', []), max_sources, None)
            mask_list  = _pad_or_trim(item.get('mask', []), max_sources, 0)

            # Zapisz metadane dotyczące źródeł
            for w in range(n_full_windows):
                start = w * self.window_len
                self.windows.append({
                    'audio_path': audio_path,
                    'start': start,
                    'azimuths': azimuths,
                    'elevations': elevations,
                    'mask': mask_list,
                    'n_sources': item.get('n_sources', 0)
                })

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        win = self.windows[idx]
        audio_path = win['audio_path']
        start = win['start']

        # Wczytaj fragment audio (jedno okno czasowe)
        waveform, fs = torchaudio.load(audio_path, frame_offset=start, num_frames=self.window_len)
        if fs != self.sr:
            waveform = torchaudio.functional.resample(waveform, fs, self.sr)
        waveform = (waveform - waveform.mean()) / (waveform.std() + 1e-8)

        # Sprawdź dBFS na oknie – jeśli poniżej progu, szukaj kolejnego niepustego okna
        window_dbfs = dbfs(waveform.clone().detach())
        if window_dbfs < self.dbfs_threshold:
            return self.__getitem__((idx + 1) % len(self.windows))

        # Przygotuj etykiety pozycji (stały rozmiar: max_sources x 3)
        az_list = win['azimuths']  # lista: np. [0, None, ...]
        el_list = win['elevations']
        mask = torch.tensor(win['mask'], dtype=torch.float32)   # np. [1, 0, 0, ...] (długość: max_sources)
        pos_labels = torch.zeros((self.max_sources, 3), dtype=torch.float32)

        # Wypełnij pozycje dla aktywnych źródeł
        for i, (az, el, m) in enumerate(zip(az_list, el_list, mask)):
            if m > 0.5 and az is not None and el is not None:
                pos_labels[i] = torch.tensor(angles_to_unitvec(az, el), dtype=torch.float32)
            # Padding (mask==0 lub None): zostaje [0,0,0]

        # Liczba źródeł (scalar, float tensor)
        n_sources = torch.tensor(float(mask.sum().item()), dtype=torch.float32)
        # Można też: n_sources = torch.tensor(win['n_sources'], dtype=torch.float32)

        # Zwróć: sygnał, liczba źródeł, maska aktywności, pozycje (wszystko już w odpowiednich wymiarach)
        return waveform.clone().detach(), n_sources, mask, pos_labels

# ---------- Pomocnicze funkcje ----------

def angles_to_unitvec(azimuth, elevation):
    """
    Zamienia kąty [azimuth, elevation] (w stopniach) na wektor [x, y, z].
    """
    if azimuth is None or elevation is None:
        return [0.0, 0.0, 0.0]
    az = np.deg2rad(azimuth)
    el = np.deg2rad(elevation)
    x = np.cos(el) * np.cos(az)
    y = np.cos(el) * np.sin(az)
    z = np.sin(el)
    return [x, y, z]

def unitvec_to_angles(vec):
    """
    Zamienia tensor [N, 3] na kąty (azimuth, elevation) w stopniach.
    """
    x, y, z = vec[:, 0], vec[:, 1], vec[:, 2]
    az = torch.atan2(y, x)
    el = torch.asin(z / torch.sqrt(x**2 + y**2 + z**2))
    return torch.rad2deg(az), torch.rad2deg(el)

def dbfs(waveform):
    """
    Oblicza poziom sygnału w dBFS (zakładając waveform [-1,1]).
    """
    rms = (waveform ** 2).mean()
    if rms == 0:
        return -120.0
    return 10 * torch.log10(rms + 1e-10)

def _pad_or_trim(seq, length, pad_value):
    """
    Dostosowuje listę do zadanego length (używane dla mask/az/el z manifestu).
    """
    seq = list(seq)
    if len(seq) < length:
        seq = seq + [pad_value] * (length - len(seq))
    elif len(seq) > length:
        seq = seq[:length]
    return seq
