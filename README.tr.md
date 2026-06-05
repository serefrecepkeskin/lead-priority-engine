# Lead Priority Engine

> For English: [`README.md`](README.md)

Satış adaylarını (lead) önceliklendiren iki parçalı bir sistemdir: tabular bir model dönüşüm olasılığını tahmin eder, bir duygu / niyet sınıflandırıcısı ise etkileşim metinleri üzerinde çalışır. Her iki sinyal birleştirilerek, satış temsilcisinin doğrudan üzerinde hareket edebileceği tek bir öncelik skoru üretilir.

Durum: 0'dan 5'e kadar olan tüm fazlar tamamlanmıştır — sentetik duygu eğitim verisi ve sızıntı (leakage) tanılaması, EDA + ortak özellik (feature) hattı, LR taban modeli + LightGBM lead skorlaması, OpenRouter LLM duygu sınıflandırıcısı, birleşik öncelik skoru ve FastAPI servisi + Docker dağıtımı. İnceleme için hazırdır.

## Proje nedir

Çalışma anında devreye giren proje **`src/lead_priority/`** paketidir — kurulan, konteynerlenen ve servis edilen kısım budur. Ağaçtaki diğer her şey destekleyici materyaldir: `notebooks/` ve `docs/` bu noktaya nasıl gelindiğini anlatır, `scripts/datagen/` ile `scripts/build_*`, `scripts/train_*`, `scripts/fit_*` yardımcı betikleri sentetik notları, eğitilmiş artefaktları ve docx yazımlarını bir kez (çevrimdışı, her istek için tekrar değil) üretmiştir, `data/` girdileri tutar, `artifacts/` ise çalışma anında yüklenen eğitilmiş modelleri içerir.

## Hızlı başlangıç

Tek komutla etkileşimli kurulum (yalnızca standart kütüphane kullanılır, Python 3.12 dışında ön koşul yoktur):

    python3 scripts/setup.py

Şablondan `.env` kopyalanır, eksikse `OPEN_ROUTER_API_KEY` istenir, Docker imajı (veya venv) inşa edilir ve `/healthz` + `/score` üzerinde sigara testi (smoke test) çalıştırılır. Herhangi bir adım hata verirse [`docs/6_deployment.docx`](docs/6_deployment.docx) §9'daki manuel kurtarma tablosuna yönlendirilir.

## Kurulum

Python 3.12 gereklidir.

    make install-dev
    cp .env.example .env   # ardından gerektiği gibi düzenleyin

`install-dev` komutu `.venv` oluşturur, bağımlılıkları kurar, paketi düzenlenebilir (editable) modda yükler ve pre-commit kancalarını (hooks) kaydeder.

## Günlük komutlar

    make lint        # ruff check
    make format      # ruff format + autofix
    make typecheck   # src/ üzerinde mypy strict
    make test        # pytest
    make pre-commit  # tüm kancaları tüm ağaç üzerinde çalıştır
    make run         # FastAPI dev sunucusu

## CI

GitHub Actions her push ve pull request'te ruff + mypy + pytest çalıştırır. İş akışı `.github/workflows/` altındadır; canlı çalıştırmalar [github.com/SerefRecepKeskin/lead-priority-engine/actions](https://github.com/SerefRecepKeskin/lead-priority-engine/actions) adresinden görülebilir — kodu çekmeden önce mevcut commit üzerinde pipeline'ın yeşil olduğunu doğrulamanın en kolay yoludur.

## Yerleşim

    src/lead_priority/    çalışma zamanı paketi (api, features, models, utils, settings)
    scripts/              CLI giriş noktaları
    scripts/datagen/      çevrimdışı veri üretim modülleri (çalışma zamanının dışında)
    tests/                pytest test takımı
    notebooks/            EDA ve deneyler
    data/                 ham girdiler + data/synthetic/ (LLM tarafından üretilmiş notlar)
    artifacts/            eğitilmiş modeller + sızıntı raporu (özetler ve joblib dosyaları)
    docs/                 yazılı çıktılar (deliverable)
    .github/workflows/    CI (ruff + mypy + pytest)

## Çalışma anında `artifacts/` klasöründen ne yüklenir

FastAPI uygulaması açılışta **hiçbir eğitim yapmaz** — yalnızca yükler. Yükleyiciler [`src/lead_priority/api/deps.py`](src/lead_priority/api/deps.py) içindedir ve süreç ömrü boyunca LRU önbellekte tutulur.

- `feature_pipeline.joblib` — uydurulmuş (fitted) özellik dönüştürücüsü (`get_feature_transformer` tarafından yüklenir)
- `lead_scoring_lgbm.joblib` — eğitilmiş LightGBM lead skorlama modeli (dosya adı `LEAD_SCORING_MODEL` ortam değişkeniyle değiştirilebilir)
- `sentiment_predictions/glm-4-5-air_test.parquet` — `/leads/top` önbelleğini inşa etmek için kullanılan, önceden hesaplanmış duygu etiketleri; böylece `/leads/top` her istekte LLM çağırmaz
- `lead_scoring_metrics.json`, `sentiment_metrics.json`, `priority_metrics.json` — `/readyz` üzerinden yüzeye çıkarılır (yalnızca başlık metrikleri; dosyalar yoksa hata verilmez)

Yeniden eğitim notebooklarda ve çevrimdışı `scripts/train_*`, `scripts/fit_*` yardımcılarındadır — istek yolunda asla yapılmaz.

## Yapılandırma

Çalışma zamanı yapılandırması `.env` dosyasında (gitignore'da) bulunur ve `src/lead_priority/settings.py` tarafından `pydantic-settings` aracılığıyla yüklenir. `.env.example` repoya işlenmiş şablondur.

### LLM API anahtarları — hangisi ne içindir

Repo iki LLM sağlayıcısına atıf yapar. İkisinin rolleri tamamen farklıdır ve servisi çalıştıracak bir incelemecinin hangisinin gerçekten gerekli olduğunu bilmesi gerekir:

- **OpenRouter (`OPEN_ROUTER_API_KEY`)** — **çalışma anında kullanılan tek LLM**. `/score` uç noktasındaki Faz 3 duygu sınıflandırmasını besler. Tanımlı değilse `/score` nötr duyguya geri düşer ve servis kullanılabilir kalır; `/readyz` eksik anahtarı 503 olarak yüzeye çıkarır.
- **Azure OpenAI (`AZURE_OPENAI_*`)** — **yalnızca çevrimdışı veri üretimi için** kullanılır (Faz 0 sentetik etkileşim notları, `scripts/datagen/` ve `scripts/generate_interactions.py` tarafından üretilir). Çalışma zamanı paketi bunu hiç içe aktarmaz. Üretilen notlar zaten `data/synthetic/` altında işlenmiş durumda olduğundan, servisi çalıştıracak bir incelemecinin Azure anahtarına **ihtiyacı yoktur**.

## Belgeler

Her fazın kendi yazımı `docs/` altında, varsa keşifsel notebook'u ise `notebooks/` altındadır. Dosyalar numaralandırılmıştır, böylece kronolojik sıra bir bakışta görülür. README kasıtlı olarak kısa tutulmuştur — derinlik için ilgili belgeye tıklayın.

| # | Faz | Yazım | Notebook |
|---|---|---|---|
| 0 | Sentetik etkileşim verisi + sızıntı tanılaması | [`docs/0_synthetic_data_and_leakage.docx`](docs/0_synthetic_data_and_leakage.docx) | [`notebooks/0_leakage_analysis.ipynb`](notebooks/0_leakage_analysis.ipynb) |
| 1 | EDA + özellik mühendisliği | [`docs/1_eda_and_feature_engineering.docx`](docs/1_eda_and_feature_engineering.docx) | [`notebooks/1_eda_and_feature_engineering.ipynb`](notebooks/1_eda_and_feature_engineering.ipynb) |
| 2 | Lead skorlama modeli (LR taban + LGBM) | [`docs/2_lead_scoring.docx`](docs/2_lead_scoring.docx) | [`notebooks/2_lead_scoring.ipynb`](notebooks/2_lead_scoring.ipynb) |
| 3 | Duygu / niyet sınıflandırıcısı (OpenRouter LLM zero/few-shot) | [`docs/3_sentiment_classifier.docx`](docs/3_sentiment_classifier.docx) | [`notebooks/3_sentiment_classifier.ipynb`](notebooks/3_sentiment_classifier.ipynb) |
| 4 | Birleşik öncelik skoru (ağırlıklı ortalama) | [`docs/4_priority_score.docx`](docs/4_priority_score.docx) | [`notebooks/4_priority_demo.ipynb`](notebooks/4_priority_demo.ipynb) |
| 6 | FastAPI servisi + Docker dağıtımı (servis tasarımı + kurulum rehberi) | [`docs/6_deployment.docx`](docs/6_deployment.docx) | — |

**Belge format sözleşmesi** (numaralı her docx aynı biçimi izler):

- En üstte tıklanabilir içindekiler tablosu
- Üçüncü tekil şahıs / edilgen Türkçe — yazar için değil, incelemeci için yazılmıştır
- Teknik kavramlar önce sade dille, sonra biçimsel notasyonla açıklanır
- Bölüm numaralandırması `1. → 1.1 → 1.2 → 2.` şeklindedir
- Sayısal sonuçlar metin içinde değil, tablolarda verilir

**Notebook sözleşmesi:**

- Numaralı önek karşılık gelen docx ile eşleşir
- Kod hücreleri arasındaki markdown hücreleri bir sonraki bloğun *ne* yaptığını ve *neden* yaptığını açıklar — incelemeci kodu çalıştırmadan baştan sona okuyabilmelidir
- İlk markdown hücresi faz docx'ine (`docs/N_*.docx`) geri bağlantı verir

## Dağıtım (Deployment)

Kurulum, servis tasarımı, hata-modu sorun giderme tablosu ve manuel kurtarma adımları için **[`docs/6_deployment.docx`](docs/6_deployment.docx)** dosyasına bakın. Tek komutluk `scripts/setup.py` mutlu yolu kapsar; docx ise geri kalan her şeyi kapsar (Docker ve venv modları, uç nokta tanımları, bir adım hata verdiğinde kurulum betiğinin atıf yaptığı kurtarma prosedürleri).

## Vaka çalışmasının servis yüzeyini kapsayan testler

Vaka çalışması FastAPI servisini değerlendirir. Aşağıdaki dört test bu yüzeyin çalıştığını ispatlar; `tests/` altındaki diğer testler model ve özellik katmanları için destekleyici testlerdir.

| Dosya | Ne ispatlar |
|---|---|
| [`tests/test_api_health.py`](tests/test_api_health.py) | `/healthz` canlılık + `/readyz` model-yüklendi kontrolleri, request-ID middleware |
| [`tests/test_api_logging.py`](tests/test_api_logging.py) | Yapılandırılmış JSON logların request-id ilişkilendirmesi ve istisna yığını (stack) ile basılması |
| [`tests/test_api_score.py`](tests/test_api_score.py) | `POST /score` mutlu yolu, istek doğrulama, OpenRouter erişilemez / rate-limit olduğunda nötr duyguya zarif düşüş |
| [`tests/test_api_top_leads.py`](tests/test_api_top_leads.py) | `GET /leads/top` sıralama, sayfalama (`n`), `min_priority` filtresi, önceden hesaplanmış açılış önbelleğinden sunum |

`make test` ile çalıştırılır. `tests/` altındaki diğer testler model katmanını (lead scoring, sentiment, priority) ve özellik hattını kapsar — destekleyici testlerdir, vaka çalışmasının servis yüzeyi değil.

## Proje ağacı

```
lead-priority-engine/
├── src/lead_priority/          ← asıl proje (kurulan, konteynerlenen, servis edilen)
│   ├── api/                    FastAPI uygulaması
│   │   ├── main.py             uygulama factory + lifespan (açılışta modelleri ısıtır)
│   │   ├── deps.py             artifacts/ klasöründen okunan her şey için LRU önbellekli yükleyiciler
│   │   ├── schemas.py          Pydantic istek / yanıt modelleri
│   │   ├── errors.py           istisna işleyicileri (OpenRouter, yapılandırma, doğrulama)
│   │   ├── logging.py          JSON formatlayıcı + request-id middleware
│   │   └── endpoints/
│   │       ├── health.py       /healthz, /readyz
│   │       ├── score.py        POST /score (tek lead için birleşik öncelik)
│   │       └── top_leads.py    GET /leads/top (önceden hesaplanmış önbellek)
│   ├── features/               özellik hattı (derive + transformers + kalıcılık)
│   ├── models/
│   │   ├── lead_scoring.py     LR / LightGBM sarmalayıcı
│   │   ├── sentiment.py        OpenRouter LLM duygu sınıflandırıcı
│   │   └── priority.py         ağırlıklı ortalama öncelik formülü
│   ├── utils/                  küçük ortak yardımcılar
│   └── settings.py             pydantic-settings yükleyicisi (.env)
├── tests/                      pytest takımı (API testleri için yukarıdaki Testler bölümüne bakın)
├── artifacts/                  çalışma anında okunan eğitilmiş modeller + metrikler
│   ├── feature_pipeline.joblib
│   ├── lead_scoring_lgbm.joblib
│   ├── lead_scoring_lr.joblib                              (LR taban modeli; varsayılan olarak yüklenmez)
│   ├── sentiment_predictions/glm-4-5-air_test.parquet      (/leads/top önbellek kaynağı)
│   ├── lead_scoring_metrics.json / sentiment_metrics.json / priority_metrics.json
│   ├── feature_summary.json / leakage_report.json          (Faz 0–1 tanılamaları)
│   └── figures/                docx yazımlarına gömülen grafikler
├── data/
│   ├── Lead Scoring.csv        ham girdi
│   ├── Leads Data Dictionary.xlsx
│   └── synthetic/              repoya işlenmiş LLM-üretimi etkileşim notları (Faz 0)
├── docs/                       numaralı faz yazımları (0–4, 6) — Belgeler tablosuna bakın
├── notebooks/                  belge numaralarıyla eşleşen EDA + deneyler
├── scripts/
│   ├── setup.py                tek komutluk etkileşimli kurulum (yalnızca standart kütüphane)
│   ├── datagen/                çevrimdışı Faz-0 sentetik veri araçları (çalışma zamanı DEĞİL)
│   ├── generate_interactions.py / evaluate_openrouter_sentiment.py
│   ├── fit_feature_pipeline.py / train_lead_scoring.py    (çevrimdışı eğitim)
│   └── build_*_docx.py / build_3_sentiment_notebook.py    (docx + notebook üreticileri)
├── examples/score_request.json örnek POST /score yükü
├── Dockerfile                  çok aşamalı build (yalnızca çalışma zamanı)
├── Makefile                    install-dev / lint / format / typecheck / test / run
├── .env.example                yapılandırma şablonu (src/lead_priority/settings.py yükler)
└── pyproject.toml
```
