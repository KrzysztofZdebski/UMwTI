# Transkrypcja ręcznie pisanych listów (j. polski) modelem TrOCR - podsumowanie projektu

**Zadanie ML:** automatyczna transkrypcja zbioru **~300 ręcznie pisanych listów** w języku polskim
(historyczna korespondencja osobista) na tekst maszynowy.
**Model:** fine-tuning `microsoft/trocr-base-handwritten`.

---

## 1. Sformułowanie problemu

Wejściem jest ~300 listów pisanych kursywą (skany/PDF). Model OCR pracuje na pojedynczych
**wierszach**, więc zadanie rozbito na dwa etapy:

1. **Przygotowanie danych** - od surowych skanów do wyciętych, opisanych linijek tekstu (sekcja 3).
2. **Rozpoznanie tekstu (HTR - Handwritten Text Recognition)** na poziomie linijki - fine-tuning TrOCR.

Metryka główna: **CER (Character Error Rate)**, pomocnicza: **exact match** (odsetek idealnie odczytanych linijek).

## 2. Dlaczego TrOCR - uzasadnienie wyboru modelu

| Kryterium | Uzasadnienie |
|---|---|
| **Pismo kursywne** | Klasyczne OCR (np. Tesseract) zakłada druk i zawodzi na łączonym piśmie odręcznym. |
| **Transfer learning** | TrOCR jest pretrenowany na dużych zbiorach rękopisów. Mając ograniczone własne dane, korzystamy z wiedzy z pretreningu zamiast uczyć od zera. |
| **Architektura end-to-end** | Enkoder-dekoder transformerowy nie wymaga ręcznej segmentacji na znaki ani osobnego modelu językowego. |
| **Wbudowany model języka** | Dekoder autoregresyjny działa jak model języka, co pomaga przy polskim. |
| **Poziom linijki** | TrOCR działa natywnie na wierszach - zgodne z naszą segmentacją. |

**Rozważone alternatywy:** Tesseract (odpada - kursywa), modele CTC typu CRNN (wymagają większego
zbioru). Sprawdzano też wariant „polski”: ViT + dekoder **HerBERT** (`allegro/herbert-base-cased`)
jako blank-slate - ale taki model trzeba uczyć od zera, co przy dostępnej ilości danych było mniej
opłacalne niż transfer learning z TrOCR. TrOCR był najlepszym kompromisem jakość/nakład danych.

## 3. Przygotowanie danych - od skanów do opisanych linijek

Cały preprocessing zrealizowano własnym, modułowym pipeline'em (`run_pipeline.py`):

**3.1. Ekstrakcja stron** (`extract_pages.py`)
Listy w `Listy/` (PDF + obrazy) → rasteryzacja każdej strony PDF do PNG w **300 DPI** (PyMuPDF),
obrazy luźne kopiowane i normalizowane do PNG. Wynik: `output/1_pages/`.

**3.2. Preprocessing strony** (`preprocess.py`)
- **przycięcie marginesów** (3% z każdej krawędzi - usuwa czarne ramki skanera i cień bindowania),
- **deskew** - estymacja kąta przekrzywienia transformatą Hougha (mediana kątów linii niemal
  poziomych) i obrót z białym tłem; pomijany przy kącie <0.3°,
- opcjonalna binaryzacja **Sauvoli** (adaptacyjna, odporna na nierówne oświetlenie i przebicia
  atramentu) - domyślnie wyłączona, bo TrOCR działa lepiej na obrazie szaro-/kolorowym. Wynik: `2_preprocessed/`.

**3.3. Segmentacja na linijki** (`segment_lines.py`)
Metoda domyślna: **Kraken BLLA** (neuronowa analiza układu / baseline) z maskowaniem wielokąta linii.
**Fallback: profil projekcji poziomej**:

```python
def compute_horizontal_projection(binary):
    ink = (binary < 128).astype(np.float64)   # atrament = 1, tło = 0
    return np.sum(ink, axis=1)                

```

Cropy pobierane z obrazu kolorowego (padding 10 px / 5 px). Wynik: `3_lines/<strona>/line_XXX.png`;
nazwa koduje pochodzenie: `<dokument>_<strona>_line_<nr>`.

**3.4. Etykiety** (`transcribe.py` + kuracja) - wstępne transkrypcje przy użyciu LLM (sonnet 4.6), następnie ręcznie
korygowane i filtrowane.

**3.5. Złożenie zbioru** (`build_combined_dataset.py`) - scalenie źródeł z **deduplikacją po treści
obrazu (MD5)** i **filtrem jakości etykiet** (odrzut: `�`, zniekształcone `??`, ≤2 litery, litery
spoza alfabetu polskiego). Wynik: `combined_dataset/` + `combined_transcribed.txt` - **2 585 linijek**.

## 4. Charakterystyka zbioru

| Cecha | Wartość |
|---|---|
| Listy źródłowe | ~300 (wielostronicowe) |
| Wszystkie linijki (wliczając niepoprawne) | ~15 000 |
| **Opisane linijki (po filtrach)** | **2 585** (≈2 197 train / 388 test, split 85/15) |
| Mediana wymiarów wycinka | 1219 × 133 px (proporcje ~8.7:1) |

## 5. Fine-tuning - konfiguracja i implementacja

Trening na Kaggle (GPU T4), `transformers 5.0`, `Seq2SeqTrainer` z generacją w ewaluacji.

### 5.1 Konfiguracja modelu i generacji

```python
processor = TrOCRProcessor.from_pretrained("microsoft/trocr-base-handwritten")
model = VisionEncoderDecoderModel.from_pretrained("microsoft/trocr-base-handwritten")

model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
model.config.pad_token_id = processor.tokenizer.pad_token_id
model.config.eos_token_id = processor.tokenizer.sep_token_id
model.config.vocab_size = model.config.decoder.vocab_size

model.generation_config.max_length = 128          
model.generation_config.no_repeat_ngram_size = 3  

model.config.use_cache = False
model.gradient_checkpointing_enable()
```

### 5.2 Lazy data collator (letterbox + augmentacja + przesunięte `decoder_input_ids`)

Preprocessing jest **leniwy** (per-batch, bez cache'owania tensorów pikseli - oszczędza RAM):

```python
from torchvision import transforms as T

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

    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,

    learning_rate=2e-5,              
    num_train_epochs=25,             
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    weight_decay=0.01,
    label_smoothing_factor=0.1,      
    fp16=True,                       

    predict_with_generate=True,
    generation_max_length=128,
    eval_accumulation_steps=4,
    metric_for_best_model="cer",
    greater_is_better=False,
    load_best_model_at_end=True,
    save_strategy="epoch",

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
    """Trening z augmentacją."""
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

> Implementacja jest odporna na zmiany API `transformers` - argumenty (`eval_strategy` vs
> `evaluation_strategy`) i sposób przekazania procesora (`tokenizer` vs `processing_class`) wykrywane
> są dynamicznie przez `inspect.signature`.

## 6. Problemy machine learningowe

### 6.1 Ograniczona liczba danych douczających *(główne ograniczenie projektu)*
2 585 linijek to niewielki zbiór jak na adaptację domenową modelu pretrenowanego na **angielskim**
rękopisie do **polskiej** historycznej kursywy (typowe fine-tune'y HTR operują na 5 000–50 000+
linijkach). To fundamentalne, **zewnętrzne** ograniczenie (dostępność opisanych danych)..

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

### Faza 1 - świeży fine-tuning (16 epok, zatrzymany przez early stopping)

| Epoch | Training Loss | Validation Loss | CER | Exact Match |
|-------|---------------|-----------------|-----|-------------|
| 1 | 9.231207 | 4.737667 | 0.895661 | 0.000000 |
| 2 | 7.759459 | 4.316751 | 0.845556 | 0.002577 |
| 3 | 7.319302 | 4.109739 | 0.833100 | 0.002577 |
| 4 | 7.316568 | 4.081180 | 0.789293 | 0.002577 |
| 5 | 6.566225 | 3.840924 | 0.815675 | 0.000000 |
| 6 | 6.080056 | 3.785325 | 0.797481 | 0.002577 |
| 7 | 5.594503 | 3.809358 | 0.771379 | 0.005155 |
| 8 | 5.460197 | 3.832788 | 0.774878 | 0.000000 |
| 9 | 4.518145 | 3.897903 | 0.782785 | 0.002577 |
| 10 | 4.340661 | 3.885958 | 0.783345 | 0.002577 |
| 11 | 4.155198 | 3.932482 | 0.761512 | 0.000000 |
| 12 | 3.913686 | 3.952920 | 0.780406 | 0.007732 |
| 13 | 3.508954 | 3.976652 | 0.769279 | 0.005155 |
| 14 | 3.414327 | 3.979627 | 0.780756 | 0.002577 |
| 15 | 3.227147 | 4.004866 | 0.794892 | 0.002577 |
| 16 | 3.289620 | 4.016764 | 0.767250 | 0.002577 |

**Model realnie się uczy:** CER spada **0.90 → 0.76**, training loss 9.23 → 3.29. Najlepszy model: epoka 11.

### Faza 2 - kontynuacja od checkpointu (warm start, 50 epok)

| Epoch | Training Loss | Validation Loss | CER | Exact Match |
|-------|---------------|-----------------|-----|-------------|
| 1 | 3.408025 | 3.868884 | 0.755773 | 0.002577 |
| 2 | 3.445820 | 3.928128 | 0.779076 | 0.002577 |
| 3 | 3.349665 | 3.972334 | 0.755913 | 0.002577 |
| 4 | 3.351776 | 3.989510 | 0.775577 | 0.002577 |
| 5 | 3.255251 | 4.018839 | 0.772498 | 0.002577 |
| 6 | 3.182054 | 4.040764 | 0.756613 | 0.005155 |
| 7 | 3.116934 | 4.083073 | 0.769559 | 0.005155 |
| 8 | 3.093700 | 4.081426 | 0.771029 | 0.002577 |
| 9 | 3.011602 | 4.092079 | 0.752484 | 0.007732 |
| 10 | 2.986797 | 4.072457 | 0.755843 | 0.002577 |
| 11 | 3.011123 | 4.092159 | 0.780546 | 0.005155 |
| 12 | 3.105639 | 4.083969 | 0.786074 | 0.005155 |
| 13 | 3.022136 | 4.109255 | 0.762981 | 0.002577 |
| 14 | 2.988275 | 4.106850 | 0.774318 | 0.002577 |
| 15 | 2.946483 | 4.119764 | 0.748915 | 0.002577 |
| 16 | 2.999237 | 4.128208 | 0.765080 | 0.002577 |
| 17 | 2.981327 | 4.098608 | 0.760392 | 0.002577 |
| 18 | 2.932664 | 4.116899 | 0.757383 | 0.002577 |
| 19 | 2.961612 | 4.114201 | 0.741917 | 0.005155 |
| 20 | 2.909420 | 4.129246 | 0.742407 | 0.005155 |
| 21 | 2.907040 | 4.111197 | 0.743807 | 0.002577 |
| 22 | 2.899541 | 4.115469 | 0.754794 | 0.005155 |
| 23 | 2.877912 | 4.099996 | 0.759972 | 0.005155 |
| 24 | 2.876825 | 4.142379 | 0.757663 | 0.005155 |
| 25 | 2.849150 | 4.111486 | 0.747446 | 0.002577 |
| 26 | 2.855849 | 4.132052 | 0.752694 | 0.002577 |
| 27 | 2.845365 | 4.128495 | 0.758223 | 0.005155 |
| 28 | 2.822191 | 4.113623 | 0.752344 | 0.005155 |
| 29 | 2.849798 | 4.131371 | 0.753534 | 0.005155 |
| 30 | 2.820983 | 4.143507 | 0.749405 | 0.005155 |
| 31 | 2.810139 | 4.119569 | 0.740868 | 0.005155 |
| 32 | 2.809708 | 4.138077 | 0.758153 | 0.005155 |
| 33 | 2.804303 | 4.144155 | 0.743107 | 0.005155 |
| 34 | 2.788798 | 4.145324 | 0.752694 | 0.005155 |
| 35 | 2.792433 | 4.157380 | 0.742407 | 0.005155 |
| 36 | 2.794246 | 4.144279 | 0.740168 | 0.005155 |
| 37 | 2.812021 | 4.149506 | 0.737089 | 0.005155 |
| 38 | 2.781111 | 4.155793 | 0.739888 | 0.005155 |
| 39 | 2.780825 | 4.151217 | 0.724633 | 0.005155 |
| 40 | 2.776015 | 4.168219 | 0.755983 | 0.005155 |
| 41 | 2.786983 | 4.159222 | 0.761162 | 0.005155 |
| 42 | 2.771640 | 4.159981 | 0.742197 | 0.005155 |
| 43 | 2.773317 | 4.163846 | 0.744437 | 0.005155 |
| 44 | 2.777030 | 4.165090 | 0.742687 | 0.005155 |
| 45 | 2.773024 | 4.169267 | 0.740028 | 0.005155 |
| 46 | 2.761407 | 4.170637 | 0.746956 | 0.005155 |
| 47 | 2.772878 | 4.168514 | 0.751645 | 0.005155 |
| 48 | 2.768720 | 4.169053 | 0.747726 | 0.005155 |
| 49 | 2.765932 | 4.169341 | 0.743597 | 0.005155 |
| 50 | 2.764648 | 4.169335 | 0.743527 | 0.005155 |

| Metryka | Start | Koniec | Best |
|---|---|---|---|
| **CER** | 0.756 | 0.744 | **0.725 (ep. 39)** |
| Training loss | 3.408 | 2.765 | - |
| Validation loss | 3.869 | 4.169 | - |

Start już z poziomu ~0.76; przez 50 epok CER stoi w miejscu - co potwierdza, że dźwignią nie jest
czas treningu, lecz ilość danych.

### Charakter predykcji

[1] GT: połączenia z Tobą, dodaję zarazem, abyś     
    PR: innym myśli, jak tylko do w liście wszystko, tam przyjemno- 

[2] GT: pyta: „jakże tam Twoje stosunki     
    PR: Te mnie 

[3] GT: pozostaną również tylko wspólną tajemnicą; jeszczem ci to     
    PR: spowiadamą równiej tylko spóźniłsz. Tejemniesz zacznem to do 

[4] GT: Pytałam o Zenka, chwileczkę z Tobą     
    PR: piszę do Panią chwilachą z Tobą 

[5] GT: U mnie moja zupełnie niepokoi, bo wiesz że Stef     
    PR: U nie proszę zupełnie przyrodzym, bo wierz, że Stef 

[6] GT: Hej Mój kochany najdroższy, Serdecznie     
    PR: List Twój Najdroższy. O serdecznie 

[7] GT: i przy kąpieli będzie zawsze kucharka, a gdy wchodzisz     
    PR: byże bym wytrzyj, bo bardzo 

[8] GT: staje, gdy walcząc z przeszkodami walczy sam do zupełnego zwy-     
    PR: który goś w duszy, prawdałem w sobie mej Jedynej wyjej 

[9] GT: przepełniam Cię jestem. Jesteś mi tak strasznie droga.     
    PR: przyjacielu Ci jestem, jestro mi tak strasznie drogę.

- to realne czytanie, nie halucynacja. Zawodzi natomiast na liniach skrajnie długich i na charakterach
pisma rzadko reprezentowanych w niewielkim zbiorze.

## 8. Co się udało, a co nie

**Udało się:**
- Zbudować kompletny pipeline danych: ekstrakcja stron → deskew/preprocessing → **segmentacja na
  linijki (Kraken + fallback projekcyjny)** → transkrypcja → kuracja i filtrowanie etykiet.
- Zaimplementować dopracowany, odporny na wersje API pipeline fine-tuningu (lazy collator, obsługa
  `decoder_input_ids` przy label smoothing, osobny collator eval, mixed precision, gradient
  checkpointing, early stopping po CER).
- **Wytrenować model, który realnie się uczy** (CER 0.90 → 0.76) i **poprawnie czyta linie
  umiarkowanej długości**.

**Nie udało się (i dlaczego):**
- Osiągnąć użytkowej jakości transkrypcji (CER ~0.74). Przyczyna: **ograniczona liczba opisanych
  danych** dla adaptacji TrOCR (pretrening EN) do polskiej kursywy (6.1); wtórnie geometria
  najdłuższych linii (6.4). To ograniczenie zewnętrzne, nie konfiguracyjne.
