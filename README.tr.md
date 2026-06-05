# Lead Priority Engine

> For English: [`README.md`](README.md)

Satış adaylarını (lead) önceliklendiren iki parçalı bir sistemdir: tabular bir model dönüşüm olasılığını tahmin eder, bir duygu / niyet sınıflandırıcısı ise etkileşim metinleri üzerinde çalışır. Her iki sinyal birleştirilerek, satış temsilcisinin doğrudan üzerinde hareket edebileceği tek bir öncelik skoru üretilir.

Durum: 0'dan 5'e kadar olan tüm fazlar tamamlanmıştır — sentetik duygu eğitim verisi ve sızıntı (leakage) tanılaması, EDA + ortak özellik (feature) hattı, LR taban modeli + LightGBM lead skorlaması, OpenRouter LLM duygu sınıflandırıcısı, birleşik öncelik skoru ve FastAPI servisi + Docker dağıtımı. İnceleme için hazırdır.

## Proje nedir

Çalışma anında devreye giren proje **`src/lead_priority/`** paketidir — kurulan, konteynerlenen ve servis edilen kısım budur. Ağaçtaki diğer her şey destekleyici materyaldir: `notebooks/` ve `docs/` bu noktaya nasıl gelindiğini anlatır, `scripts/train_*` ve `scripts/fit_*` yardımcı betikleri eğitilmiş artefaktları bir kez (çevrimdışı, her istek için tekrar değil) üretmiştir, `data/` girdileri tutar, `artifacts/` ise çalışma anında yüklenen eğitilmiş modelleri içerir.

## Hızlı başlangıç

Tek komutla etkileşimli kurulum (yalnızca standart kütüphane kullanılır, Python 3.12 dışında ön koşul yoktur):

    python3 deploy/setup.py

Şablondan `.env` kopyalanır, eksikse `OPEN_ROUTER_API_KEY` istenir, Docker imajı (veya venv) inşa edilir ve `/healthz` + `/score` üzerinde sigara testi (smoke test) çalıştırılır. Herhangi bir adım hata verirse [`docs/5_fastapi_serving_and_deployment.docx`](docs/5_fastapi_serving_and_deployment.docx) §9'daki manuel kurtarma tablosuna yönlendirilir.

### OpenRouter API anahtarını alma (ücretsiz — kredi kartı gerekmez)

Çalışma zamanı **`z-ai/glm-4.5-air:free`** modelini OpenRouter üzerinden kullanır; bu model **ücretsiz katmandadır** (free tier). Projeyi çalıştırmak için **herhangi bir ödeme yöntemi eklemek veya kredi bakiyesi yüklemek gerekmez**. Ücretsiz katmanın günlük istek limiti değerlendirme için fazlasıyla yeterlidir.

1. **<https://openrouter.ai/>** adresine gidin ve kayıt olun (Google / GitHub / e-posta — hepsi ücretsiz).
2. Anahtarlar sayfasını açın: **<https://openrouter.ai/keys>**.
3. **Create Key** butonuna tıklayın, herhangi bir isim verin (örn. `lead-priority-engine`) ve değeri kopyalayın (`sk-or-...` ile başlar).
4. `.env` dosyasında `OPEN_ROUTER_API_KEY=` satırına yapıştırın (veya `python3 deploy/setup.py` çalıştırırsanız o sizden ister).

İşlem bu kadar — fatura ayarı, kredi yükleme yok. Günlük kota dolarsa `/score` otomatik olarak nötr sentiment'a düşer ve servis çalışmaya devam eder; ertesi gün tekrar deneyebilirsiniz.

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

    src/lead_priority/    çalışma zamanı paketi (api, core, infra, utils, settings)
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

- **OpenRouter (`OPEN_ROUTER_API_KEY`)** — **çalışma anında kullanılan tek LLM**. `/score` uç noktasındaki Faz 3 duygu sınıflandırmasını besler. Yapılandırılmış `z-ai/glm-4.5-air:free` modeli OpenRouter'ın **ücretsiz katmanındadır** — fatura ayarı gerekmez. Anahtarı nasıl alacağınız için yukarıdaki [OpenRouter API anahtarını alma](#openrouter-api-anahtarını-alma-ücretsiz--kredi-kartı-gerekmez) bölümüne bakın. Tanımlı değilse `/score` nötr duyguya geri düşer ve servis kullanılabilir kalır; `/readyz` eksik anahtarı 503 olarak yüzeye çıkarır.
- **Azure OpenAI (`AZURE_OPENAI_*`)** — **yalnızca çevrimdışı veri üretimi için** kullanılır (Faz 0 sentetik etkileşim notları, `scripts/datagen/` ve `scripts/generate_interactions.py` tarafından üretilir). Çalışma zamanı paketi bunu hiç içe aktarmaz. Üretilen notlar zaten `data/synthetic/` altında işlenmiş durumda olduğundan, servisi çalıştıracak bir incelemecinin Azure anahtarına **ihtiyacı yoktur**.

## Belgeler

Her fazın kendi yazımı `docs/` altında, varsa keşifsel notebook'u ise `notebooks/` altındadır. Dosyalar numaralandırılmıştır, böylece kronolojik sıra bir bakışta görülür. README kasıtlı olarak kısa tutulmuştur — derinlik için ilgili belgeye tıklayın.

### Faz 0 — Sentetik veri + sızıntı tanılaması

Sentetik etkileşim notları (TR / EN / Mix code-switching), etiketleme stratejisi, sentetik ↔ ham join üzerinde train→serve sızıntı tanılaması.

📄 [`docs/0_synthetic_data_and_leakage.docx`](docs/0_synthetic_data_and_leakage.docx) · 📓 [`notebooks/0_leakage_analysis.ipynb`](notebooks/0_leakage_analysis.ipynb)

### Faz 1 — EDA + özellik mühendisliği

Dönüşüm oranı dağılımı, sınıf dengesizliği, eksik veri paterni, kaynak bazında dönüşüm farkları; türetilen özellikler (`channel_diversity_count`, `total_time_per_visit`, `days_since_last_activity`, …) her birinin gerekçesiyle.

📄 [`docs/1_eda_and_feature_engineering.docx`](docs/1_eda_and_feature_engineering.docx) · 📓 [`notebooks/1_eda_and_feature_engineering.ipynb`](notebooks/1_eda_and_feature_engineering.ipynb)

### Faz 2 — Lead skorlama modeli

LR taban (yorumlanabilirlik) vs LightGBM (modern, hyperparameter-tuned); ROC / PR / accuracy + calibration plot + threshold sweep + top-%20 gain & lift chart; bootstrap-CI paired test; SHAP feature önemi.

📄 [`docs/2_lead_scoring.docx`](docs/2_lead_scoring.docx) · 📓 [`notebooks/2_lead_scoring.ipynb`](notebooks/2_lead_scoring.ipynb)

### Faz 3 — Duygu / niyet sınıflandırıcısı

Dört attitude (`positive_engagement` / `objection` / `neutral` / `disengaged`); OpenRouter açık kaynak LLM, zero/few-shot prompt (XLM-R / DistilBERT fine-tune alternatifi tartışıldı); TR + EN + Mix dil desteği; sınıf-bazlı + dil-bazlı confusion matrix + macro-F1 + bootstrap CI; fairness ve etik analizi.

📄 [`docs/3_sentiment_classifier.docx`](docs/3_sentiment_classifier.docx) · 📓 [`notebooks/3_sentiment_classifier.ipynb`](notebooks/3_sentiment_classifier.ipynb)

### Faz 4 — Birleşik öncelik skoru

`P(conversion)` + sentiment ordinal ağırlıklı ortalama; ağırlık gerekçesi, sensitivity sweep, meta-model alternatifi bilinçli olarak elendi.

📄 [`docs/4_priority_score.docx`](docs/4_priority_score.docx) · 📓 [`notebooks/4_priority_demo.ipynb`](notebooks/4_priority_demo.ipynb)

### Faz 5 — FastAPI servisi + Docker dağıtımı

`POST /score` + `GET /leads/top` endpoint sözleşmeleri, yapılandırılmış JSON logging + request-ID middleware, çok aşamalı Dockerfile, integration testleri, manuel kurtarma tablosu; üretim notları — feature drift takibi, retrain sıklığı, satış temsilcisi geri bildirim döngüsü, false-positive maliyet çerçevesi, 3-gün bütçesi sonraki adımlar.

📄 [`docs/5_fastapi_serving_and_deployment.docx`](docs/5_fastapi_serving_and_deployment.docx) · (notebook yok)

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

## API örnekleri

> İnteraktif Swagger UI: **http://127.0.0.1:8000/docs** — `POST /score` üzerindeki “Try it out” butonu tam dolu bir örnek payload ile önceden doldurulmuştur; JSON elle yazmadan gerçek bir istek atabilirsiniz. Redoc da `/redoc` adresinde mevcut.

`POST /score` — tek bir lead için birleşik priority skoru. İstek yükü `examples/score_request.json`:

```bash
$ curl -X POST http://localhost:8000/score \
       -H "Content-Type: application/json" \
       -d @examples/score_request.json | jq .
{
  "p_conversion": 0.7234,
  "sentiment": {
    "predicted_attitude": "objection",
    "sentiment_score": 0.65,
    "sentiment_unavailable": false,
    "latency_ms": 412.5
  },
  "priority": 0.6940,
  "weights": { "weight_conversion": 0.6, "weight_sentiment": 0.4 },
  "model_versions": { "feature_pipeline_schema": 2, "lead_scoring_kind": "lightgbm", "sentiment_model_name": "z-ai/glm-4.5-air:free" },
  "request_id": "5f2e1c3a..."
}
```

`GET /leads/top?n=N` — birleşik priority'ye göre sıralı top-N lead. Açılışta inşa edilen in-memory cache'ten servis edilir (istek başına LLM çağrısı **yok**):

```bash
$ curl 'http://localhost:8000/leads/top?n=3' | jq .
{
  "count": 3,
  "total_available": 924,
  "leads": [
    { "lead_id": "74878c4b-...", "p_conversion": 0.994828, "predicted_attitude": "positive_engagement", "sentiment_score": 1.0, "priority": 0.996897, "language": "tr" },
    { "lead_id": "2caa32d0-...", "p_conversion": 0.994314, "predicted_attitude": "positive_engagement", "sentiment_score": 1.0, "priority": 0.996588, "language": "tr" },
    { "lead_id": "bd5ca024-...", "p_conversion": 0.993130, "predicted_attitude": "positive_engagement", "sentiment_score": 1.0, "priority": 0.995878, "language": "en" }
  ],
  "model_versions": { "feature_pipeline_schema": 2, "lead_scoring_kind": "lightgbm", "sentiment_model_name": "z-ai/glm-4.5-air:free" },
  "request_id": "d3baf801..."
}
```

`min_priority` ve `n` (en fazla 924) desteklenen query parametreleri; daha küçük `n` aynı sıralı listenin baştan bir dilimini döner.

## Dağıtım (Deployment)

Kurulum, servis tasarımı, hata-modu sorun giderme tablosu ve manuel kurtarma adımları için **[`docs/5_fastapi_serving_and_deployment.docx`](docs/5_fastapi_serving_and_deployment.docx)** dosyasına bakın. Tek komutluk `deploy/setup.py` mutlu yolu kapsar; docx ise geri kalan her şeyi kapsar (Docker ve venv modları, uç nokta tanımları, bir adım hata verdiğinde kurulum betiğinin atıf yaptığı kurtarma prosedürleri).

## Servis yüzeyini kapsayan testler

FastAPI servisi üretimde çalışan yüzeydir. Aşağıdaki dört test bu yüzeyin uçtan uca çalıştığını ispatlar; `tests/` altındaki diğer testler model ve özellik katmanları için destekleyici testlerdir.

| Dosya | Ne ispatlar |
|---|---|
| [`tests/test_api_health.py`](tests/test_api_health.py) | `/healthz` canlılık + `/readyz` model-yüklendi kontrolleri, request-ID middleware |
| [`tests/test_api_logging.py`](tests/test_api_logging.py) | Yapılandırılmış JSON logların request-id ilişkilendirmesi ve istisna yığını (stack) ile basılması |
| [`tests/test_api_score.py`](tests/test_api_score.py) | `POST /score` mutlu yolu, istek doğrulama, OpenRouter erişilemez / rate-limit olduğunda nötr duyguya zarif düşüş |
| [`tests/test_api_top_leads.py`](tests/test_api_top_leads.py) | `GET /leads/top` sıralama, sayfalama (`n`), `min_priority` filtresi, önceden hesaplanmış açılış önbelleğinden sunum |

`make test` ile çalıştırılır. `tests/` altındaki diğer testler model katmanını (lead scoring, sentiment, priority) ve özellik hattını kapsar — destekleyici testlerdir, ana servis yüzeyi değil.

## Proje ağacı

```
lead-priority-engine/
├── src/lead_priority/          ← asıl proje (kurulan, konteynerlenen, servis edilen)
│   ├── api/                    FastAPI uygulaması (transport katmanı)
│   │   ├── main.py             uygulama factory + lifespan (açılışta modelleri ısıtır)
│   │   ├── deps.py             artifacts/ klasöründen okunan her şey için LRU önbellekli yükleyiciler
│   │   ├── schemas.py          Pydantic istek / yanıt modelleri
│   │   ├── errors.py           istisna işleyicileri (OpenRouter, yapılandırma, doğrulama)
│   │   ├── middleware.py       request-id + JSON erişim-log middleware
│   │   └── endpoints/
│   │       ├── health.py       /healthz, /readyz
│   │       ├── score.py        POST /score (tek lead için birleşik öncelik)
│   │       └── top_leads.py    GET /leads/top (önceden hesaplanmış önbellek)
│   ├── core/                   domain mantığı (transport yok, dış IO yok)
│   │   ├── features/           özellik hattı (derive + transformers + kalıcılık)
│   │   ├── inference/lead_scoring.py   LR / LightGBM sarmalayıcı
│   │   └── scoring/
│   │       ├── priority.py     ağırlıklı ortalama öncelik formülü
│   │       └── sentiment_classes.py  SentimentClass + etiket-skor map'i
│   ├── infra/                  dış servisler için adaptörler
│   │   └── openrouter/sentiment.py   OpenRouter LLM duygu sınıflandırıcı
│   ├── utils/
│   │   └── logging.py          JSON formatlayıcı + rotating dosya handler kurulumu
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
├── docs/                       numaralı faz yazımları (0–5) — Belgeler tablosuna bakın
├── notebooks/                  belge numaralarıyla eşleşen EDA + deneyler
├── scripts/
│   ├── datagen/                çevrimdışı Faz-0 sentetik veri araçları (çalışma zamanı DEĞİL)
│   ├── generate_interactions.py / evaluate_openrouter_sentiment.py
│   └── fit_feature_pipeline.py / train_lead_scoring.py    (çevrimdışı eğitim)
├── deploy/
│   └── setup.py                tek komutluk etkileşimli kurulum (yalnızca standart kütüphane)
├── logs/                       rotating JSON servis logları (gitignore'da)
├── examples/score_request.json örnek POST /score yükü
├── Dockerfile                  çok aşamalı build (yalnızca çalışma zamanı)
├── Makefile                    install-dev / lint / format / typecheck / test / run
├── .env.example                yapılandırma şablonu (src/lead_priority/settings.py yükler)
└── pyproject.toml
```
