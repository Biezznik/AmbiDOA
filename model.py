import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from AmbiDOA import log_print           # Twoja funkcja logująca (ładne printy z timestampem)
from datetime import datetime


class SinusoidalPositionalEncoding(nn.Module):
    """
    Pozycjonowanie sinusoidalne (bezuczące się, klasyczne z prac o Transformerze).
    Dlaczego to jest potrzebne?
      - Transformery nie „wiedzą” same z siebie, w jakiej kolejności leci sygnał w czasie.
      - Pozycjonalka „dodaje” informację o indeksie próbki w sekwencji.
    Jak to działa?
      - Dla każdej pozycji (t) oraz wymiaru cechy (d) generujemy sin/cos o różnych częstotliwościach.
      - Otrzymujemy tablicę [L, d_model], którą dodaje się do wektora wejściowego.
    """
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        # pe: przygotowujemy macierz pozycji (max_len x d_model)
        pe = torch.zeros(max_len, d_model)            # [L, d_model]
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # [L,1]

        # Wzór z oryginalnego Transformera (Vaswani et al.):
        # różne częstotliwości dla kanałów parzystych i nieparzystych
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )  # [d_model/2]

        # Kanały parzyste -> sin, nieparzyste -> cos
        pe[:, 0::2] = torch.sin(position * div_term)   # kolumny 0,2,4,...
        pe[:, 1::2] = torch.cos(position * div_term)   # kolumny 1,3,5,...
        pe = pe.unsqueeze(0)                           # dodaj wymiar batcha: [1, L, d_model]

        # register_buffer -> to NIE są parametry uczące się,
        # ale będą przenoszone wraz z modelem (na GPU/CPU)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, L, d_model]
        Zwracamy x + wycinek pozycjonalki [1, L, d_model].
        """
        L = x.size(1)
        return x + self.pe[:, :L, :]


class AmbiTransformerDOA(nn.Module):
    """
    Transformer do lokalizacji wielu źródeł w ambisonics:
      - HEAD 1 (count): przewidywana liczba źródeł (regresja -> 1 wartość na batch)
      - HEAD 2 (positions): pozycje [max_sources, 3] jako wektory jednostkowe (x,y,z) na sferze

    Architektura wysokopoziomowo:
      1) Projekcja kanałów (16) -> d_model (np. 256): "poszerzamy" opis chwili w czasie.
      2) (opcjonalnie) dodajemy pozycjonalkę czasową (żeby Transformer „znał” porządek).
      3) Przepuszczamy przez kilka warstw Encoder-a Transformera.
      4) Pooling po czasie (średnia) -> jeden wektor na cały fragment czasowy.
      5) Dwie głowy (małe MLP) -> osobno przewidywanie liczby źródeł i pozycji.

    Słowniczek:
      - d_model: wymiar „wewnętrznego” wektora w Transform erze. To NIE jest długość sygnału.
                 d_model = ile liczb opisuje „stan” w danej chwili (po projekcji z 16 kanałów).
      - nhead:   na ile „głów” (niezależnych mechanizmów uwagi) dzielimy d_model.
      - num_layers: ile warstw Encoder-a (głębokość).
      - dim_feedforward: rozmiar wewnętrznych MLP w każdej warstwie Encoder-a (po attention).
    """
    def __init__(
        self,
        n_channels: int = 16,       # liczba kanałów wejściowych (u Ciebie 16)
        input_len:   int = 2756,    # długość okna w próbkach — tu tylko do długości pozycjonalki
        d_model:     int = 256,     # wymiar modelu po projekcji (np. 128/256/384...)
        nhead:       int = 8,       # liczba głów attention; d_model musi być podzielne przez nhead
        num_layers:  int = 8,       # liczba warstw encoder-a (głębokość)
        dim_feedforward: int = 1024,# rozmiar MLP w warstwie encoder-a (większe = większa pojemność)
        dropout:     float = 0.3,   # regularizacja
        max_sources: int = 12,      # ile maksymalnie źródeł opisujemy na wyjściu
        use_posenc:  bool = True,   # czy użyć pozycjonalki
    ):
        super().__init__()
        assert d_model % nhead == 0, "d_model musi być podzielne przez nhead"
        self.max_sources = max_sources

        # (1) Projekcja 16 -> d_model.
        #    Każda próbka w czasie ma wektor 16 wartości (kanały). Zamieniamy to na bogatszy
        #    opis d_model-wymiarowy (np. 256), dodając też normalizację warstwową dla stabilności.
        self.proj = nn.Sequential(
            nn.Linear(n_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # (2) Pozycjonalka (dodajemy „informację o czasie” do każdej próbki).
        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len=max(4096, input_len)) \
                       if use_posenc else nn.Identity()

        # (3) Główny Encoder Transformera (uwaga wielogłowa + MLP).
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,    # wejścia mają postać [B, T, d_model]
            norm_first=True,     # „Pre-LN” – zwykle stabilniejszy trening
            activation="gelu",   # aktywacja w MLP (często lepsza niż ReLU)
        )
        self.transformer   = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.pre_head_norm = nn.LayerNorm(d_model)  # normalizacja przed głowami wyjściowymi



        # (4) HEAD 1: przewidywanie liczby źródeł (regresja 1D).
        # Uwaga: wejście do headów to [B, d_model], więc używamy MLP (bez RNN).
        self.n_sources_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1)
        )

        # (5) HEAD 2: przewidywanie pozycji (max_sources * 3 wartości -> (x,y,z) dla każdego slotu).
        self.positions_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, max_sources * 3),
        )

    def forward(self, x: torch.Tensor):
        """
        x: [batch, n_channels, n_samples]  -> [B, 16, T]
        Zwraca:
          n_sources:  [B, 1]              (liczba źródeł w oknie)
          positions:  [B, max_sources, 3] (wektory jednostkowe na sferze)
        """
        # Zamiana osi, bo Transformer pracuje na [B, T, C] (czas pośrodku).
        x = x.permute(0, 2, 1)          # [B, T, C]  (T = n_samples, C = n_channels)

        # Projekcja 16 -> d_model (bogatsza reprezentacja chwili czasu).
        x = self.proj(x)                 # [B, T, d_model]

        # Dodanie informacji o pozycji w sekwencji (czas).
        x = self.pos_enc(x)              # [B, T, d_model]

        # Przepuszczenie przez warstwy Transformera (self-attention + MLP).
        feats = self.transformer(x)      # [B, T, d_model]

        # Pooling po osi czasu – średnia „reprezentacja” całego okna.
        feats = feats.mean(dim=1)        # [B, d_model]

        # Drobna normalizacja przed głowami – często stabilizuje trening.
        feats = self.pre_head_norm(feats)

        # HEAD 1: liczba źródeł (1 liczba na przykład).
        n_sources = self.n_sources_head(feats)               # [B, 1]

        # HEAD 2: pozycje wszystkich „slotów” (S = max_sources).
        positions = self.positions_head(feats)               # [B, S*3]
        positions = positions.view(-1, self.max_sources, 3)  # [B, S, 3]

        # Normalizacja do wektorów jednostkowych (na sferę), żeby x^2 + y^2 + z^2 ≈ 1
        positions = positions / (positions.norm(dim=2, keepdim=True) + 1e-8)

        return n_sources, positions
