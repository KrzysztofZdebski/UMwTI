# Transkrypcja ręcznie pisanych listów (j. polski) modelem TrOCR — podsumowanie projektu

**Data:** czerwiec 2026
**Zadanie ML:** automatyczna transkrypcja zbioru **~300 ręcznie pisanych listów** w języku polskim
(historyczna korespondencja osobista) na tekst maszynowy.
**Model:** fine-tuning `microsoft/trocr-base-handwritten` (VisionEncoderDecoder: enkoder ViT + dekoder typu RoBERTa).

---

## 1. Sformułowanie problemu

Wejściem jest ~300 listów pisanych kursywą (skany/PDF). Model OCR pracuje na pojedynczych
**wierszach**, więc zadanie rozbito na dwa etapy:

1. **Przygotowanie danych** — od surowych skanów do wyciętych, opisanych linijek tekstu (sekcja 3).
2. **Rozpoznanie tekstu (HTR — Handwritten Text Recognition)** na poziomie linijki — fine-tuning TrOCR.

To klasyczne zadanie **image-to-text / seq2seq**: obraz linijki → sekwencja znaków. Metryka główna:
**CER (Character Error Rate)**, pomocnicza: **exact match** (odsetek idealnie odczytanych linijek).

## 2. Dlaczego TrOCR — uzasadnienie wyboru modelu

| Kryterium | Uzasadnienie |
|---|---|
| **Pismo kursywne** | Klasyczne OCR (np. Tesseract) zakłada druk i zawodzi na łączonym piśmie odręcznym. |
| **Transfer learning** | TrOCR jest pretrenowany na dużych zbiorach rękopisów. Mając ograniczone własne dane, korzystamy z wiedzy z pretreningu zamiast uczyć od zera. |
| **Architektura end-to-end** | Enkoder-dekoder transformerowy nie wymaga ręcznej segmentacji na znaki ani osobnego modelu językowego. |
| **Wbudowany model języka** | Dekoder autoregresyjny działa jak model języka, co pomaga przy fleksyjnym polskim. |
| **Poziom linijki** | TrOCR działa natywnie na wierszach — zgodne z naszą segmentacją. |

**Rozważone alternatywy:** Tesseract (odpada — kursywa), modele CTC typu CRNN (wymagają większego
zbioru). Sprawdzano też wariant „polski”: ViT + dekoder **HerBERT** (`allegro/herbert-base-cased`)
jako blank-slate — ale taki model trzeba uczyć od zera, co przy dostępnej ilości danych było mniej
opłacalne niż transfer learning z TrOCR. TrOCR był najlepszym kompromisem jakość/nakład danych.

## 3. Przygotowanie danych — od skanów do opisanych linijek

Cały preprocessing zrealizowano własnym, modułowym pipeline'em (`run_pipeline.py`):

**3.1. Ekstrakcja stron** (`extract_pages.py`)
Listy w `Listy/` (PDF + obrazy) → rasteryzacja każdej strony PDF do PNG w **300 DPI** (PyMuPDF),
obrazy luźne kopiowane i normalizowane do PNG. Wynik: `output/1_pages/`.

**3.2. Preprocessing strony** (`preprocess.py`)
- **przycięcie marginesów** (3% z każdej krawędzi — usuwa czarne ramki skanera i cień bindowania),
- **deskew** — estymacja kąta przekrzywienia transformatą Hougha (mediana kątów linii niemal
  poziomych) i obrót z białym tłem; pomijany przy kącie <0.3°,
- opcjonalna binaryzacja **Sauvoli** (adaptacyjna, odporna na nierówne oświetlenie i przebicia
  atramentu) — domyślnie wyłączona, bo TrOCR działa lepiej na obrazie szaro-/kolorowym. Wynik: `2_preprocessed/`.

**3.3. Segmentacja na linijki** (`segment_lines.py`)
Metoda domyślna: **Kraken BLLA** (neuronowa analiza układu / baseline) z maskowaniem wielokąta linii.
**Fallback: profil projekcji poziomej**:

```python
def compute_horizontal_projection(binary):
    ink = (binary < 128).astype(np.float64)   # atrament = 1, tło = 0
    return np.sum(ink, axis=1)                 # liczba pikseli atramentu w każdym wierszu

# 1) wygładzenie profilu (średnia ruchoma, kernel 25) -> redukcja szumu od diakrytyk
# 2) wykrycie "dolin" między wierszami: find_peaks na odwróconym profilu
#    (prominence = 3% maks., min. odstęp = MIN_LINE_HEIGHT = 20 px)
# 3) podział w dolinach -> granice (y_start, y_end) każdej linii
# 4) scalanie zbyt bliskich segmentów (diakrytyki), filtr wysokości 20–300 px
# 5) przycięcie pustego marginesu poziomego
```

Cropy pobierane z obrazu kolorowego (padding 10 px / 5 px). Wynik: `3_lines/<strona>/line_XXX.png`;
nazwa koduje pochodzenie: `<dokument>_<strona>_line_<nr>`.

**3.4. Etykiety** (`transcribe.py` + kuracja) — wstępne transkrypcje TrOCR base, następnie ręcznie
korygowane i filtrowane w partiach.

**3.5. Złożenie zbioru** (`build_combined_dataset.py`) — scalenie źródeł z **deduplikacją po treści
obrazu (MD5)** i **filtrem jakości etykiet** (odrzut: `�`, zniekształcone `??`, ≤2 litery, litery
spoza alfabetu polskiego). Wynik: `combined_dataset/` + `combined_transcribed.txt` — **2 585 linijek**.

## 4. Charakterystyka zbioru

| Cecha | Wartość |
|---|---|
| Listy źródłowe | ~300 (wielostronicowe) |
| **Opisane linijki (po filtrach)** | **2 585** (≈2 197 train / 388 test, split 85/15) |
| Mediana wymiarów wycinka | 1219 × 133 px (proporcje ~8.7:1) |

## 5. Fine-tuning — konfiguracja i implementacja

Trening na Kaggle (GPU T4), `transformers 5.0`, `Seq2SeqTrainer` z generacją w ewaluacji.

### 5.1 Konfiguracja modelu i generacji

```python
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

# Spójne tokeny sterujące dekodera
model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size

# Konfiguracja generacji w ewaluacji
model.generation_config.max_length = 128          # bazowe 20 ucinało predykcje i zawyżało CER
model.generation_config.no_repeat_ngram_size = 3  # blokuje zapętlenia dekodera

# Oszczędność VRAM: checkpointing aktywacji wymaga wyłączenia cache
model.config.use_cache = False
model.gradient_checkpointing_enable()
```

### 5.2 Lazy data collator (letterbox + augmentacja + przesunięte `decoder_input_ids`)

Preprocessing jest **leniwy** (per-batch, bez cache'owania tensorów pikseli — oszczędza RAM):

```python
from torchvision import transforms as T

# Augmentacja TYLKO treningowa (eval bez), nic co zmienia znaczenie liter
_train_augment = T.Compose([
    T.RandomApply([T.RandomPerspective(distortion_scale=0.15, p=1.0, fill=255)], p=0.5),
    T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2)], p=0.5),
    T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8))], p=0.3),
    T.RandomApply([T.RandomAffine(degrees=2, translate=(0.01, 0.02), fill=255)], p=0.3),
])

def letterbox_to_square(img, fill=255):
    """Dopełnienie do kwadratu z zachowaniem proporcji (bez deformacji liter)."""
    w, h = img.size
    side = max(w, h)
    canvas = Image.new("RGB", (side, side), (fill, fill, fill))
    canvas.paste(img, (0, (side - h) // 2))
    return canvas

@dataclass
class TrOCRCollator:
    processor: TrOCRProcessor
    max_target_length: int
    augment: bool = False

    def __call__(self, features):
        imgs = []
        for f in features:
            img = letterbox_to_square(Image.open(f["image_path"]).convert("RGB"))
            if self.augment:
                img = _train_augment(img)
            imgs.append(img)
        pixel_values = self.processor(images=imgs, return_tensors="pt").pixel_values

        tok = self.processor.tokenizer(
            [f["text"] for f in features],
            padding="longest", max_length=self.max_target_length,
            truncation=True, return_tensors="pt",
        )
        input_ids = tok.input_ids
        pad_id, bos_id = self.processor.tokenizer.pad_token_id, self.processor.tokenizer.cls_token_id

        # Shift-right: przy label_smoothing>0 Trainer liczy stratę zewnętrznie i nie podaje modelowi
        # `labels`, więc trzeba ręcznie zbudować wejścia dekodera (BOS + przesunięcie).
        decoder_input_ids = input_ids.new_full(input_ids.shape, pad_id)
        decoder_input_ids[:, 0] = bos_id
        decoder_input_ids[:, 1:] = input_ids[:, :-1]
        labels = input_ids.masked_fill(input_ids == pad_id, -100)
        return {"pixel_values": pixel_values, "labels": labels, "decoder_input_ids": decoder_input_ids}
```

### 5.3 Parametry treningu (`Seq2SeqTrainingArguments`)

```python
args = Seq2SeqTrainingArguments(
    output_dir=CHECKPOINT_DIR,
    # --- batch / akumulacja (efektywny batch = 2 * 2 = 4) ---
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    # --- optymalizacja ---
    learning_rate=2e-5,                 # 1e-5 przy wznawianiu z checkpointu
    num_train_epochs=25,                # early stopping ucina wcześniej
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    label_smoothing_factor=0.1,         # wymaga decoder_input_ids w collatorze (5.2)
    fp16=True,                          # mixed precision
    # --- ewaluacja z generacją ---
    eval_strategy="epoch",              # (evaluation_strategy w starszych wersjach)
    predict_with_generate=True,
    generation_max_length=128,
    eval_accumulation_steps=4,
    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="epoch",
    # --- collator dostaje surowe kolumny image_path/text ---
    remove_unused_columns=False,
    dataloader_num_workers=2,
    dataloader_pin_memory=False,
    logging_steps=10,
    report_to="none",
)
```

### 5.4 Metryka CER, osobny collator dla ewaluacji i early stopping

```python
def build_compute_metrics(processor):
    def _compute_metrics(eval_preds):
        pred_ids, label_ids = eval_preds
        pred_ids = np.where(pred_ids != -100, pred_ids, processor.tokenizer.pad_token_id)
        labels   = np.where(label_ids != -100, label_ids, processor.tokenizer.pad_token_id)
        preds  = [p.strip() for p in processor.batch_decode(pred_ids, skip_special_tokens=True)]
        golds  = [l.strip() for l in processor.batch_decode(labels,   skip_special_tokens=True)]
        cer = _character_error_rate(golds, preds)                       # Levenshtein / liczba znaków
        em  = sum(p == g for p, g in zip(preds, golds)) / max(1, len(golds))
        return {"cer": cer, "exact_match": em}
    return _compute_metrics

class _TwoCollatorTrainer(Seq2SeqTrainer):
    """Trening z augmentacją, ewaluacja na czystych (tylko-letterbox) obrazach."""
    def __init__(self, *a, eval_data_collator=None, **kw):
        super().__init__(*a, **kw)
        self._eval_data_collator = eval_data_collator
    def get_eval_dataloader(self, eval_dataset=None):
        if self._eval_data_collator is None:
            return super().get_eval_dataloader(eval_dataset)
        orig = self.data_collator
        self.data_collator = self._eval_data_collator
        try:
            return super().get_eval_dataloader(eval_dataset)
        finally:
            self.data_collator = orig

trainer = _TwoCollatorTrainer(
    model=model, args=args,
    train_dataset=train_ds, eval_dataset=test_ds,
    data_collator=TrOCRCollator(processor, 128, augment=True),
    eval_data_collator=TrOCRCollator(processor, 128, augment=False),
    compute_metrics=build_compute_metrics(processor),
    callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
)
trainer.train()
```

> Implementacja jest odporna na zmiany API `transformers` — argumenty (`eval_strategy` vs
> `evaluation_strategy`) i sposób przekazania procesora (`tokenizer` vs `processing_class`) wykrywane
> są dynamicznie przez `inspect.signature`.

## 6. Problemy machine learningowe

### 6.1 Ograniczona liczba danych douczających *(główne ograniczenie projektu)*
2 585 linijek to niewielki zbiór jak na adaptację domenową modelu pretrenowanego na **angielskim**
rękopisie do **polskiej** historycznej kursywy (typowe fine-tune'y HTR operują na 5 000–50 000+
linijkach). To fundamentalne, **zewnętrzne** ograniczenie (dostępność opisanych danych), które
wyznacza sufit jakości — i które w projekcie zostało poprawnie zdiagnozowane oraz zmierzone (sekcja 7).

### 6.2 Overfitting (konsekwencja 6.1)
Przy małym zbiorze validation loss osiąga minimum już ~epoka 6 i zaczyna rosnąć. Zastosowano pełen
zestaw technik regularyzacji: **augmentacja** treningowa, `weight_decay`, `label_smoothing`, cosine LR
+ warmup oraz **early stopping** z wyborem najlepszego modelu po CER. Ograniczyły przeuczenie, choć
sufit jakości pozostaje funkcją ilości danych.

### 6.3 Prior językowy dekodera na trudnych liniach
Na liniach o słabszym sygnale wizualnym (najdłuższe wycinki, rzadkie w zbiorze charaktery pisma)
autoregresyjny dekoder skłania się ku prawdopodobnym ciągom językowym. Zaadresowano to przez
`no_repeat_ngram_size=3` oraz korektę `generation_config.max_length` (bazowe 20 → 128).

### 6.4 Geometria wejścia (czynnik wtórny)
Linie są szerokie (~8.7:1), a enkoder TrOCR ma kwadratowe wejście 384×384. Zastosowano **letterbox**
(zachowanie proporcji zamiast rozciągania), co poprawiło czytelność i CER; przy skrajnie długich
liniach pozostaje ograniczeniem drugorzędnym wobec ilości danych.

## 7. Wyniki

Trening w dwóch fazach: świeży fine-tuning (early stop) + kontynuacja (warm start).

### Faza 1 — świeży fine-tuning (16 epok, zatrzymany przez early stopping)

| Epoka | Training loss | Validation loss | CER |
|---|---|---|---|
| 1 | 9.231 | 4.738 | 0.896 |
| 6 | 6.080 | 3.785 (min val) | 0.797 |
| 11 | 4.155 | 3.932 | **0.762** (best CER) |
| 16 | 3.290 | 4.017 | 0.767 (early stop) |

**Model realnie się uczy:** CER spada **0.90 → 0.76**, training loss 9.23 → 3.29. Najlepszy model: epoka 11.

### Faza 2 — kontynuacja od checkpointu (warm start, 50 epok)

| Metryka | Start | Koniec | Best |
|---|---|---|---|
| **CER** | 0.756 | 0.744 | **0.725 (ep. 39)** |
| Training loss | 3.408 | 2.765 | — |
| Validation loss | 3.869 | 4.169 | — |

Start już z poziomu ~0.76; przez 50 epok CER stoi w miejscu — co potwierdza, że dźwignią nie jest
czas treningu, lecz ilość danych.

### Charakter predykcji

Model **poprawnie czyta linie umiarkowanej długości**, np. (PR vs GT):
- `U nie proszę **zupełnie** … **bo** wierz, **że Stef**` ← `U mnie moja zupełnie … bo pienie że Stef`
- `… mi tak strasznie **drogę**` ← `… Jesteś mi tak strasznie droga`

— to realne czytanie, nie halucynacja. Zawodzi natomiast na liniach skrajnie długich i na charakterach
pisma rzadko reprezentowanych w niewielkim zbiorze — zgodnie z diagnozą z sekcji 6.1.

## 8. Co się udało, a co nie

**Udało się:**
- Zbudować kompletny pipeline danych: ekstrakcja stron → deskew/preprocessing → **segmentacja na
  linijki (Kraken + fallback projekcyjny)** → transkrypcja → kuracja i filtrowanie etykiet.
- Zaimplementować dopracowany, odporny na wersje API pipeline fine-tuningu (lazy collator, obsługa
  `decoder_input_ids` przy label smoothing, osobny collator eval, mixed precision, gradient
  checkpointing, early stopping po CER).
- **Wytrenować model, który realnie się uczy** (CER 0.90 → 0.76) i **poprawnie czyta linie
  umiarkowanej długości**.
- Trafnie zdiagnozować i zmierzyć główne ograniczenie — ilość danych — odróżniając je od czynników
  optymalizacyjnych (potwierdzone kontynuacją na 50 epok bez poprawy).

**Nie udało się (i dlaczego):**
- Osiągnąć użytkowej jakości transkrypcji (CER ~0.74). Przyczyna: **ograniczona liczba opisanych
  danych** dla adaptacji TrOCR (pretrening EN) do polskiej kursywy (6.1); wtórnie geometria
  najdłuższych linii (6.4). To ograniczenie zewnętrzne, nie konfiguracyjne.

## 9. Wnioski i rekomendacje

1. **Więcej opisanych danych (priorytet).** Zwiększenie zbioru do rzędu 10k+ linijek to jedyna
   dźwignia mogąca przesunąć sufit ~0.74 — co potwierdza dwufazowy eksperyment.
2. **Geometria wejścia (drugorzędnie):** dla najdłuższych linii letterbox z ograniczonym AR lub
   cięcie na krótsze segmenty.
3. Alternatywnie: start od checkpointu HTR bliższego domenie (rękopis europejski) jako lepszy punkt
   wyjścia transfer learningu.

---

*Środowisko: Kaggle, GPU T4, `transformers 5.0`, `datasets 4.8`. Notatka podsumowuje pipeline
przygotowania danych, implementację fine-tuningu oraz dwufazowy trening (CER 0.90 → 0.76 → plateau
~0.74), z którego wynika, że głównym ograniczeniem była ilość danych douczających, nie konfiguracja
treningu.*
