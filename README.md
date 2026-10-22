# Emosiya Pilotu

Call center üçün **pilot/test məqsədli** lokal alət: operatorun webcam görüntüsündən üz ifadəsini
təsnif edir və modellərin **real dəqiqliyini** operatorun öz cavabları ilə müqayisə edərək ölçür.
Bir neçə modeli ([EmotiEffLib](https://github.com/sb-ai-lab/EmotiEffLib), DeepFace, Hugging Face ViT,
Kaggle CNN) **eyni kadrda yan-yana** işlədib fərqlərini ölçə bilir.

> Bu, məhsul deyil — qiymətləndirmə alətidir. Model **daxili hissi yox, üz ifadəsini** ölçür.

---

## Mündəricat

1. [Quraşdırma](#quraşdırma)
2. [Modellər və müqayisə](#modellər-və-müqayisə)
3. [İşə salma](#işə-salma)
4. [İnterfeys](#interfeys)
5. [Necə işləyir](#necə-işləyir)
6. [Konfiqurasiya](#konfiqurasiya-configyaml)
7. [Dəqiqliyin ölçülməsi](#dəqiqliyin-ölçülməsi)
8. [Araşdırma: hansı model hansı ayarda](#araşdırma-hansı-model-hansı-ayarda)
9. [CSV formatı](#csv-formatı)
10. [Hesabat](#hesabat)
11. [Benchmark](#benchmark)
12. [Məxfilik](#məxfilik)
13. [Testlər](#testlər)
14. [Layihə strukturu](#layihə-strukturu)
15. [Məhdudiyyətlər](#məhdudiyyətlər)
16. [Lisenziya qeydi](#qeyd-lisenziya-və-istifadə)

---

## Quraşdırma

Tələblər: **Python 3.10+** (3.11 ilə yoxlanılıb), Windows 10/11, macOS və ya Linux.
Windows-da UI üçün **Microsoft Edge WebView2** lazımdır. Windows 11-də o, artıq qurulub.

```bash
python -m venv .venv
.venv\Scripts\activate            # macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

Modelləri bir dəfə endirin. Bu, layihədə şəbəkədən istifadə edən **yeganə** addımdır:

```bash
python scripts/download_models.py --test-fixture
```

Skript bunları endirir:

| Fayl | Harada saxlanılır |
|---|---|
| `face_detection_yunet_2023mar.onnx` (OpenCV Zoo) | `models/` |
| 4 emosiya modelinin ONNX çəkiləri | `~/.emotiefflib/` |
| İctimai mülkiyyətdə olan NASA portreti (yalnız testlər üçün) | `tests/fixtures/face.png` |

GPU istəyirsinizsə, `--torch` əlavə edin (aşağıda [GPU](#gpu) bölməsinə baxın).

**Əlavə (FER2013) modellər** — bir dəfəlik çevirmə üçün `torch` (CPU), `transformers`, `h5py`, `keras` lazımdır.
Tətbiq özü bunlarsız işləyir, bütün modellər ONNX kimi çalışır:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements-convert.txt
python scripts/download_models.py --extra
```

| Model | Necə gəlir |
|---|---|
| DeepFace | ONNX birbaşa endirilir (`models/deepface_emotion.onnx`, 6 MB) |
| ViT (Hugging Face) | Yoxlanılmış commit-dən yüklənir, ONNX-ə çevrilir və torch ilə nəticələr müqayisə olunur (`models/vit_fer2013.onnx`, 343 MB) |
| Kaggle CNN | Kaggle giriş tələb edir. `face_model.h5` faylını [buradan](https://www.kaggle.com/datasets/abhisheksingh016/machine-model-for-emotion-detection) özünüz endirib `models/` qovluğuna qoyun, sonra `--extra`-nı yenidən işə salın. TensorFlow-suz çevirici (`scripts/keras_h5_to_onnx.py`) onu ONNX-ə çevirir və Keras ilə nəticələri müqayisə edir. |

---

## Modellər və müqayisə

| Açar (`model.name`) | UI adı | Dataset | Sinif | Lisenziya | CPU, ms/üz* |
|---|---|---|---|---|---|
| `enet_b0_8_best_afew` (default) | enet_b0 · afew | AffectNet | 8 | Apache-2.0 kod; çəkilər qeyri-kommersiya | 12 |
| `enet_b0_8_va_mtl` | enet_b0 · VA | AffectNet | 8 + valentlik/oyanma | 〃 | 12 |
| `enet_b2_8` | enet_b2 | AffectNet | 8 | 〃 | 25 |
| `mbf_va_mtl` | mbf · VA | AffectNet | 8 + valentlik/oyanma | 〃 | 10 |
| `deepface_emotion` | DeepFace | FER2013 | 7 (Həqarət yoxdur) | MIT (DeepFace) | 1 |
| `vit_fer2013` | ViT (HF) | FER2013 | 7 | Apache-2.0 | 200 |
| `kaggle_fer_cnn` | Kaggle CNN | FER2013 | 7 | CC0 | — (fayl gözlənilir) |

\* Bu maşında, tək işləyəndə ([Benchmark](#benchmark)).

**Müqayisə rejimi.** Üz bir dəfə aşkarlanır və keyfiyyət filtrindən keçir. Eyni üz kəsiyi bütün aktiv
modellərə verilir, hər modelin öz 10 saniyəlik smoothing-i var. Rejimlər (`compare.mode` və ya panelin
yuxarısındakı seçici):

| Rejim | Nə edir |
|---|---|
| `off` (UI-da **Tək**) | Yalnız əsas model |
| `auto` (UI-da **Avto**) | `compare.models` siyahısı. Başlanğıcda hər model ölçülür. Cəmi vaxt analiz intervalının `budget_ratio` payını (default 50%, yəni 333 ms) keçirsə, ən ağır əlavə model söndürülür — **kompüterin gücü çatdığı qədər**. İş zamanı ardıcıl 5 analizdə büdcə aşılsa da eyni qayda tətbiq olunur. |
| `on` (UI-da **Hamısı**) | Siyahıdakı modellərin hamısı, büdcəyə baxmadan |

Bu maşında default üçlük (enet_b0 + DeepFace + ViT) birlikdə ~300 ms çəkir — büdcəyə sığır, amma kənarda.
Ağır yük ViT-dir: başqa proqramlar CPU-nu yükləyəndə "Avto" rejimi onu söndürə bilər. Bunu kartdakı
səbəb yazısından görəcəksiniz.

İşləyərkən, kamera dayanmadan:

- **əsas modeli dəyişmək:** kartdakı ulduz və ya başlıqdakı "Əsas" menyusu;
- **modeli yandırıb-söndürmək:** kartdakı açar.

Yeni model diskdən arxa planda yüklənir; şəbəkə bloku aktiv qalır.

**Frontiers məqaləsi** (Song, 2021, *Facial Expression Emotion Recognition Model Integrating Philosophy and
Machine Learning Theory*) metod təsviridir: Gabor ROI xüsusiyyətləri + CNN, channel attention, FER2013.
Məqalədə kod və model çəkiləri paylaşılmayıb, ona görə hazır model kimi qoşulmayıb. Qoşmaq üçün modeli
FER2013-də yenidən öyrətmək lazımdır.

---

## İşə salma

| Əmr | Nə edir |
|---|---|
| `python -m src.app` | UI + webcam (əvvəlcə razılıq ekranı) |
| `python -m src.app --video fayl.mp4` | UI, kamera əvəzinə video (real vaxt sürətində) |
| `python -m src.app --headless --video fayl.mp4` | UI-sız, video sürətli emal olunur → CSV |
| `python -m src.app --headless` | UI-sız webcam, razılıq terminalda soruşulur |
| `python -m src.app --image a.jpg b.jpg` | Tək şəkillər üzrə yoxlama (CSV yazılmır) |
| `python -m src.report` | CSV → HTML hesabat |
| `python -m src.benchmark --model all --together` | Bütün modellərin sürət/RAM ölçüsü, tək-tək və birlikdə |

Əlavə arqumentlər:

- `--model <ad>` config-dəki əsas modeli əvəz edir.
- `--compare off|auto|on` müqayisə rejimini əvəz edir. `--headless --video` ilə yazılmış test videolarını bütün modellərdə oflayn emal etmək üçün faydalıdır.
- `--config <yol>` başqa config faylı seçir.
- `-v` ətraflı log çıxarır.

---

## İnterfeys

UI Apple HIG prinsipləri ilə qurulub: aydınlıq, dərinlik, "liquid glass" kartları, sistem rəngləri, SF / Segoe Variable
tipoqrafiyası. Sistemin açıq/qaranlıq temasını izləyir. Windows 11-də pəncərə fonu real **Mica** şüşəsidir
(`ui.material`). Animasiyalar `prefers-reduced-motion` ayarına hörmət edir.

| Pəncərə | Nə göstərir |
|---|---|
| **Razılıq ekranı** | Nə ölçülür, nə saxlanılır, heç nəyin göndərilmədiyi, “Dayandır” imkanı. **Kamera seçimi** (“Dəyiş”) — kamera açılmadan, yalnız adlarla. Razılıq verilmədən kamera **açılmır**. “İmtina et” proqramı kamerasız bağlayır. |
| **Əsas panel** (geniş, ~1240×800) | **Solda:** kamera önizləməsi (üz statusu ilə), əsas modelin smoothed nəticəsi, üz statusu, gecikmə, CPU, növbəti sorğuya qalan vaxt, böyük **Dayandır** düyməsi. **Sağda:** hər aktiv model üçün kart (8 ehtimalın hamısı, gecikmə, valentlik/oyanma); "Razılaşma 3/3" göstəricisi; hesablama büdcəsi zolağı; **son 60 saniyənin zaman xətti** — hər modelin nə dediyi rənglə; aktiv olmayan və quraşdırılmamış modellər siyahısı. **Başlıqda:** kamera menyusu, əsas model menyusu və Tək/Avto/Hamısı seçicisi. Yaşıl “trafik işığı” yalnız əsas nəticəni göstərən kiçik pəncərəyə keçirir. |
| **Kamera indikatoru** | Ekranın yuxarı mərkəzində “Dynamic Island” tipli kapsul. Kamera açıq olduğu müddətdə həmişə görünür, öz **Dayandır** düyməsi var. Taskbar-da görünmür. |
| **Əhval sorğusu** | Sağ aşağı küncdə bildiriş. **Fokusu oğurlamır** (operator CRM-də yazmağa davam edir), 30 saniyədə özü bağlanır. |

Klaviatura qısayolu: əsas paneldə `Ctrl+Shift+P` sorğunu dərhal göstərir (nümayiş üçün).

**Kamera seçimi.** Bir neçə kamera qoşulubsa (məs. daxili + USB), başlıqdakı kamera menyusundan və ya razılıq
ekranındakı “Dəyiş” düyməsindən seçin:

- Siyahı Windows DirectShow-dan **adları ilə** oxunur və kameranı **açmır** — razılıqdan əvvəl də işığı yanmır.
  Menyu hər açılanda yenilənir: sonradan taxılan kamera da görünür.
- İşləyərkən seçilsə, yeni kamera əvvəlcə açılır, sonra köhnəsi bağlanır. Analiz və CSV dayanmır. Yeni kamera
  açılmasa (məs. başqa proqram istifadə edir), xəbərdarlıq çıxır və köhnə kamera işləməyə davam edir.
- Nəzarətli test gedərkən kamera dəyişdirilmir — testin bütün kadrları bir kameradan olmalıdır.
- Seçim `local_state.json`-da (ad + cihaz yolu) yadda qalır. USB kamera başqa porta taxılsa da tanınır.
  Fayl yoxdursa və ya kamera qoşulmayıbsa, `capture.camera_index` istifadə olunur.

**Dizaynı brauzerdə önizləmə.** `src/ui/web/mock.js` saxta məlumat verir:

```bash
python -m http.server 8765 --bind 127.0.0.1 --directory src/ui/web
```

Sonra `http://127.0.0.1:8765/index.html?phase=running&mtl=1` açın. Digər parametrlər: `blind=1`, `noface=1`, `compare=off|auto|on`, `prompt.html?prompt=1`.

---

## Necə işləyir

```
Kamera (arxa thread, yalnız SON kadr yaddaşda)
   │  saniyədə 1–2 dəfə (capture.analysis_fps)
   ▼
YuNet üz aşkarlama ── ən böyük üz ── keyfiyyət filtri ──✗──► "üz yoxdur" (model işə salınmır)
   │ ✓
   ▼
Kəsik (BGR→RGB) → EmotiEffLib (8 ehtimal [+ valence/arousal])
   ▼
10 saniyəlik hərəkətli ortalama → ok | qeyri-müəyyən (<0.3) | toplanır (<3 kadr)
   ▼
CSV (yalnız rəqəmlər) + UI
```

**Keyfiyyət filtri** (`detector.py`). Üz tam görünmürsə, təxmin etmirik:

| Səbəb (`face_reason`) | Şərt |
|---|---|
| `not_found` | Üz tapılmadı |
| `low_confidence` | YuNet confidence < 0.8 |
| `partial` | Üz kadrın kənarında kəsilib, yaxud landmark-lar qutudan kənardadır |
| `too_small` | Üz < 80 px (kameradan çox uzaq) |
| `profile` | Burnun göz xəttinə görə sapması > 0.35 (yanakı baxış) |
| `tilted` | Göz xəttinin bucağı > 25° |
| `occluded` | Göz/ağız ətrafında detal yoxdur (əl, mikrofon, əşya) |

Örtülmə yoxlaması niyə lazımdır: YuNet ağzı örtülmüş üzü də 0.87+ confidence ilə tapır (testlərlə yoxlanılıb).
Ona görə göz və ağız ətrafındakı gradient sıxlığı bütün üzlə müqayisə olunur. Bu, **evristikdir** — aşağıda
[Məhdudiyyətlər](#məhdudiyyətlər) bölməsinə baxın.

Arxa planda başqa üz görünsə belə, yalnız **ən böyük üz** götürülür. Hətta o pis görünəndə də kiçik üzə keçilmir.

**Smoothing** (`smoothing.py`):

- Son 10 saniyədə üzü olan kadrların ehtimalları ortalanır.
- Tək kadrın nəticəsi heç vaxt göstərilmir: pəncərədə 3-dən az kadr varsa, nəticə “toplanır” olur.
- Hazırkı kadrda üz yoxdursa, köhnə təxmin göstərilmir.

---

## Konfiqurasiya (`config.yaml`)

| Açar | Default | İzah |
|---|---|---|
| `operator_id` | `OP-7F3A` | **Anonim** ID. Ad yazmayın — validasiya rəqəm tələb edir. |
| `model.name` | `enet_b0_8_best_afew` | Əsas model — [Modellər](#modellər-və-müqayisə) cədvəlindəki istənilən açar |
| `model.engine` | `onnx` | `onnx` yalnız CPU-dur (kitabxana `CPUExecutionProvider`-ə bağlıdır). GPU üçün `torch` (yalnız EmotiEffLib). |
| `model.device` | `cpu` | `cuda` yalnız `engine: torch` ilə. CUDA yoxdursa, xəbərdarlıqla CPU-ya keçir. |
| `model.flip_tta` | `false` | Hər kadr həm də güzgü əksi ilə proqnozlaşdırılır, ehtimallar ortalanır (2× vaxt) |
| `compare.mode` | `auto` | `off` / `auto` / `on` — bax [Modellər və müqayisə](#modellər-və-müqayisə) |
| `compare.models` | enet_b0, DeepFace, ViT, Kaggle | Müqayisə ediləcək modellər. Quraşdırılmayanlar UI-da səbəbi ilə göstərilir. |
| `compare.budget_ratio` | `0.5` | `auto` rejimində bütün modellərin cəmi vaxtı analiz intervalının bu payını keçmir |
| `detector.*` | bax faylda | confidence 0.8, min üz 80 px, yanakı/əyik/örtülmə hədləri |
| `detector.crop_margin` | `0.1` | Modelə verilən kəsikdə üz qutusunun hər tərəfinə boşluq (0 = AffectNet kimi boşluqsuz) |
| `detector.align` | `false` | Kəsikdən əvvəl gözləri üfüqi et (baş əyikliyini düzəlt) |
| `capture.camera_index` | `0` | Seçim edilməyibsə və ya seçilmiş kamera qoşulmayıbsa istifadə olunan kamera (UI-dakı seçim üstündür) |
| `capture.analysis_fps` | `1.5` | Saniyədə emal olunan kadr (0.2–2). 30 FPS emal edilmir. |
| `smoothing.window_seconds` | `10` | Hərəkətli ortalama pəncərəsi |
| `smoothing.min_frames` | `3` | Nəticə göstərmək üçün minimum kadr |
| `smoothing.uncertain_threshold` | `0.3` | Ən yüksək ehtimal bundan aşağı → qeyri-müəyyən. 8 sinifdə təsadüfi səviyyə 12,5%-dir; aşağı hədd daha çox nəticə, amma daha çox səhv deməkdir (bax `python -m src.evaluate` — əminlik həddi cədvəli) |
| `self_report.interval_minutes` | `10` | Sorğu tezliyi |
| `self_report.timeout_seconds` | `30` | Avtomatik bağlanma |
| `self_report.categories` | 6 kateqoriya | `kod: ekranda görünən ad` |
| `mapping` | bax faylda | 8 model sinfi → operator kateqoriyası (hesabat üçün) |
| `ui.show_prediction` | `true` | `false` — **kor rejim** (bax [Dəqiqliyin ölçülməsi](#dəqiqliyin-ölçülməsi)) |
| `ui.show_preview` | `false` | Kamera önizləməsi (yalnız yaddaşda) |
| `ui.material` | `auto` | `auto` / `mica` / `acrylic` / `none` (Windows 11 şüşə fonu) |
| `privacy.block_network` | `true` | Modellər yükləndikdən sonra bütün xarici bağlantıları bloklayır |
| `report.min_group_size` | `3` | Daha az operator olduqda hesabatda xəbərdarlıq |
| `report.va_tired_rule` | söndürülüb | `_mtl` modellərində aşağı oyanma → “Yorğun” |
| `research.save_test_frames` | `true` | Nəzarətli testdə şəkil saxlama **imkanı**. `false` — razılıq açarı UI-da görünmür, heç nə saxlanmır |
| `research.dir` | `dataset` | Araşdırma şəkillərinin qovluğu |
| `research.crop_margin` | `0.6` | Saxlanan kəsiyin boşluğu — sonradan istənilən daha kiçik boşluğu sınamaq üçün geniş |
| `research.jpeg_quality` | `95` | |

Default mapping:

| Model sinfi | Operator kateqoriyası |
|---|---|
| Neutral | Neytral |
| Happiness | Yaxşı / Xoşbəxt |
| Sadness | Kədərli |
| Anger, Disgust, Contempt | Əsəbi |
| Fear, Surprise | Stresli |

**“Yorğun” kateqoriyasının modeldə qarşılığı yoxdur.** Hesabat onu ayrıca göstərir və əsas uyğunluq faizinə daxil etmir.

Config səhvləri işə salma anında aydın mesajla dayandırılır: naməlum açar, yanlış model, `onnx` + `cuda`,
ad kimi görünən `operator_id` və s.

---

## Dəqiqliyin ölçülməsi

Hər 10 dəqiqədən bir operatora bildiriş göstərilir: **Neytral, Yaxşı/Xoşbəxt, Yorğun, Əsəbi, Kədərli, Stresli**
və ya **“Cavab vermək istəmirəm”**. Bildiriş işə mane olmur və 30 saniyədə özü bağlanır.

Modelin nəticəsi bildirişin **göstərildiyi anda** götürülür, cavab verilən anda yox. Səbəb: pəncərənin özü üz
ifadəsini dəyişə bilər.

Hesabatdakı göstəricilər:

| Göstərici | Nə ölçür |
|---|---|
| **Uyğunluq** (əsas) | Model müəyyən nəticə verib, kateqoriyanın modeldə qarşılığı var — bu cavablarda üst-üstə düşmə faizi. Wilson 95% etibar intervalı ilə. |
| **Cohen kappa** | Təsadüfi uyğunluğu çıxır |
| **Sadə bazis** | “Həmişə ən çox seçilən cavabı desəydik” — model bundan yaxşı olmalıdır |
| **Bütün cavablar üzrə** | Qeyri-müəyyən və üz yoxdur halları uyğunsuz sayılır |
| **Modelin əminliyi** | Cavab anlarında modelin müəyyən nəticə verdiyi pay |

> ⚠️ **Qərəz riski.** Operator ekranda modelin nəticəsini görürsə, öz cavabını ona “uyğunlaşdıra” bilər.
> Qərəzsiz ölçü üçün `ui.show_prediction: false` (kor rejim) tövsiyə olunur. Bu rejimdə panel yalnız kameranın
> işlədiyini və üz statusunu göstərir, CSV isə yenə tam yazılır.

### Nəzarətli test (göstərişli ifadələr)

Sahə sorğusu operatorun **hissini** soruşur. Nəzarətli test isə modelin **ifadəni oxuma** qabiliyyətini
"düzgün cavab" məlum olan şəraitdə ölçür: hansı ifadənin göstərildiyini bilirik.

**Necə aparılır:**

1. Kamera işləyərkən başlıqdakı **"Nəzarətli test"** düyməsinə basın. Açılan pəncərədə:
   - test olunan şəxsin **anonim ID-si** (məs. `T-01`; ad qəbul edilmir — ən azı bir rəqəm tələb olunur);
   - **"Şəkilləri araşdırma üçün saxla"** açarı — yalnız həmin şəxs razı olduqda yandırın
     (bax [Araşdırma](#araşdırma-hansı-model-hansı-ayarda)). Açar hər testdə söndürülmüş açılır.
2. Ekranda ardıcıl göstərişlər çıxır, məsələn: "Gülümsəyin — yanaqlar qalxsın…". Hər addım üç mərhələdən ibarətdir:
   - **Hazırlaşın** (3 s);
   - **İndi saxlayın** (8 s);
   - **Rahatlayın** (2 s).
3. Default olaraq 7 ifadə × 2 raund (~3 dəqiqə) aparılır. Raund daxilində sıra təsadüfidir, Neytral həmişə birincidir.

**Qaydalar:**

- Yalnız **"İndi saxlayın"** mərhələsinin kadrları qiymətləndirilir və CSV-yə `record_type=test`,
  `expected=<ifadə>` kimi yazılır.
- Hər kadrın **öz nəticəsi** qiymətləndirilir, 10 saniyəlik ortalama yox. Əks halda əvvəlki ifadə növbətiyə qarışar.
- Test zamanı:
  - model nəticələri **gizlidir** — tester ifadəsini modelə "uyğunlaşdırmasın";
  - kamera önizləməsi açıq olur — üz çərçivədə qalsın;
  - sorğu bildirişləri dayanır.
- Test kadrları sahə statistikasına (keyfiyyət, saatlıq paylanma, sorğu uyğunluğu) **qarışmır**.
- Bütün modellərin ölçülməsi üçün testdən əvvəl rejimi **"Hamısı"** edin.

**Nəticə:**

- Test bitəndə ekranda hər model üçün **düzgünlük**, **balanslı düzgünlük** (ifadələrin ortalaması) və ifadələr
  üzrə heatmap görünür.
- `python -m src.report` hesabatında "Nəzarətli test" bölməsi bunlara 95% interval və ən yaxşı modelin səhv
  matrisini əlavə edir. Məsələn, səhv matrisi "Qorxu"nun əsasən "Neytral" kimi oxunduğunu göstərə bilər.

**Tövsiyə:** testi bir neçə nəfərlə, iş yerindəki real işıq və qulaqlıqla aparın.

> Pozalı ifadələr təbii ifadələrdən asandır, ona görə bu nəticə **yuxarı həddir**. Seçim üçün: nəzarətli testdə
> ifadəni oxuya bilməyən modeli çıxarın, qalanlar arasında qərarı sahə sorğusu versin.

Parametrlər `config.yaml`-ın `test:` bölməsindədir: ifadələr, raund sayı, mərhələ müddətləri, sıranın qarışdırılması.

---

## Araşdırma: hansı model hansı ayarda

Məqsəd: eyni kadrlarda bütün modelləri və ön emal ayarlarını müqayisə edib ən yaxşı kombinasiyanı tapmaq.
Canlı testdə hər dəfə yalnız bir ayar sınanır və hər tester bir dəfə gəlir — ona görə şəkillər saxlanılır,
ayarlar isə sonradan oflayn sınanır.

**1. Şəkilləri toplayın.** Nəzarətli testi "Şəkilləri saxla" açarı ilə keçirin. Saxlanılır:

- yalnız **"İndi saxlayın"** mərhələsinin kadrları (gözlənilən ifadə məlumdur);
- bütün kadr yox, **üz ətrafında kəsik** (`research.crop_margin: 0.6`);
- `dataset/<tarix>_<tester>/` qovluğunda: `meta.json` (razılıq vaxtı, plan), `frames.csv` (ifadə, addım, üz
  qutusu, 5 nöqtə), `00001_Happiness.jpg …`.

Test sonunda ekranda "N şəkil saxlanıldı" yazılır. Adi monitorinqdə heç vaxt şəkil yazılmır.

```bash
python -m src.dataset                      # sessiyalar, şəkil sayı, ölçü
python -m src.dataset --delete <sessiya>   # bir testerin şəkillərini sil
python -m src.dataset --delete-all --yes   # hamısını sil
```

**2. Müqayisə edin.**

```bash
python -m src.evaluate                     # quraşdırılmış bütün modellər × 16 ayar
python -m src.evaluate --models enet_b0_8_best_afew,enet_b0_8_va_mtl --margins 0,0.1,0.2 --align both --flip on
python -m src.evaluate --testers T-01,T-02 # yalnız bu testerlər
```

Sınanan ayarlar (canlı rejimdə eyni funksiyalar işləyir — `src/variants.py`):

| Ayar | Variantlar | Canlı config |
|---|---|---|
| Kəsik boşluğu | 0 / 10 / 20 / 30% | `detector.crop_margin` |
| Düzləndirmə (gözlər üfüqi) | söndürülü / açıq | `detector.align` |
| Flip TTA | söndürülü / açıq | `model.flip_tta` |

Üz yenidən aşkarlanmır — test vaxtı YuNet-in tapdığı qutu və nöqtələr istifadə olunur, fərq yalnız ayardan gəlir.
Proqnozlar sessiya qovluğunda keşlənir (`.cache/`, yalnız rəqəmlər): yeni tester əlavə olunanda yalnız onun
şəkilləri hesablanır.

**3. Nəticəni oxuyun.** `reports/qiymetlendirme_<tarix>.html` və eyni adlı `.csv`:

- **Ən yaxşı kombinasiya** və canlı ayarla fərq;
- **Hər model üçün ən yaxşı ayar** — addım səviyyəsində işarə testi (↑/↓ addım, p);
- **LOTO** (bir testeri kənarda saxla): ayar qalan testerlərdə seçilir, kənardakında yoxlanır. "Ən yaxşını
  seçmək" optimizmini aradan qaldırır — **yeni adamda gözlənilən faydanı** göstərir. Ən azı 3 tester lazımdır;
- **Ayarların təsiri** — hər ayarın digərləri sabit ikən orta fərqi;
- ifadələr üzrə heatmap, əminlik həddi cədvəli, bütün kombinasiyaların sıralaması.

Hesabatda tester ID-ləri yoxdur — yalnız testerlər arası yayılma. Tövsiyə olunan config **avtomatik tətbiq
olunmur**: fərq addım üzrə əhəmiyyətli və LOTO müsbət olduqda əl ilə dəyişin.

---

## CSV formatı

Fayl adı: `logs/YYYY-MM-DD_<operator_id>.csv`. Gün dəyişəndə yeni fayl yaranır. Sxem dəyişərsə, köhnə fayl
pozulmur, `_2` faylı yaranır.

Müqayisə rejimində hər analiz üçün **hər aktiv model ayrıca sətir** yazır, `model_name` fərqli, `timestamp`
eyni olur. Eyni qayda sorğu cavabına da aiddir: sorğu anında hər modelin nəticəsi ayrıca sətirdir.
FER2013 modellərində `p_contempt` boşdur, çünki bu sinif o modellərdə yoxdur.

| Sütun | İzah |
|---|---|
| `timestamp` | ISO 8601, yerli vaxt + offset, millisaniyə |
| `operator_id`, `model_name`, `engine`, `device` | Anonim ID və faktiki işləyən model/cihaz |
| `record_type` | `sample` (hər analiz kadrı), `self_report` və ya `test` (nəzarətli test kadrı, smoothing-siz) |
| `status` | `ok` / `uncertain` / `no_face` / `warming_up` |
| `face_detected`, `face_reason` | `1`/`0` və səbəb kodu ([keyfiyyət filtri](#necə-işləyir)) |
| `top_emotion`, `top_prob` | Smoothed nəticə. Qeyri-müəyyən olanda `top_emotion = uncertain`, amma ehtimallar saxlanılır. |
| `p_anger` … `p_surprise` | 8 smoothed ehtimal |
| `valence`, `arousal` | Yalnız `_mtl` modellərində |
| `n_frames` | Ortalamaya daxil olan kadr sayı |
| `self_report` | Kateqoriya kodu, `declined` və ya `timeout` |
| `latency_ms`, `detector_ms`, `model_ms` | Emal vaxtı |
| `cpu_percent` | Prosesin ümumi CPU-dan payı |
| `expected` | Yalnız `test` sətirlərində: tapşırılan ifadə (məs. `Happiness`). Bu sütundan əvvəlki fayllar da hesabatda oxunur. |

---

## Hesabat

```bash
python -m src.report                                   # logs/ → reports/hesabat_<tarix>.html
python -m src.report --from 2026-10-01 --to 2026-10-31 --model enet_b0_8_va_mtl
```

Fayl özünə-kifayətdir (xarici resurs yoxdur), açıq/qaranlıq temanı dəstəkləyir. Hər qrafikin tooltip-i və
“Cədvəl görünüşü” var. Tərkibi:

- **Əsas göstəricilər** (əsas model üzrə): uyğunluq + etibar intervalı, kappa, bazis, cavab/imtina/cavabsız
  sayları, “üz yoxdur” və “qeyri-müəyyən” payları, gecikmə, CPU.
- **Modellərin müqayisəsi** — loglarda bir neçə model varsa:
  - hər modelin operatorla uyğunluğu (zolaq + 95% interval), kappa, əminlik, qeyri-müəyyən payı, gecikmə —
    hamısı **eyni sorğu cavabları üzrə**;
  - **modellər arası razılaşma** heatmap-ı: eyni kadrda eyni kateqoriyanı demə faizi və kappa.
- **Nəzarətli test** — test aparılıbsa: hər modelin düzgünlüyü (95% interval), balanslı düzgünlük, ifadələr
  üzrə heatmap, ən yaxşı modelin səhv matrisi.
- **Confusion matrix** (əsas model): operatorun cavabı × modelin nəticəsi (+ qeyri-müəyyən / üz yoxdur
  sütunları). Kateqoriyalar üzrə recall/precision.
- **Məlumat keyfiyyəti:** status payları və “üz yoxdur” səbəbləri.
- **Gün ərzində emosiya paylanması:** saat üzrə, **yalnız qrup səviyyəsində**.
- **Performans:** orta və p95 gecikmə (detektor / model / cəmi), CPU.
- **Valentlik/oyanma** — `_mtl` modellərində, operatorun cavabına görə.

**Fərdi operator reytinqi, sıralaması və ya operator üzrə bölgü yaradılmır.** Operator ID-ləri hesabatda
ümumiyyətlə görünmür — testlə yoxlanılır.

---

## Benchmark

```bash
python -m src.benchmark --model all --together          # bütün modellər tək-tək + birlikdə, test şəkli
python -m src.benchmark --model all --camera            # webcam kadrları (razılıq soruşulur)
python -m src.benchmark --model enet_b2_8 --video f.mp4 --frames 100 --device cpu
python -m src.benchmark --model enet_b0_8_best_afew,deepface_emotion,vit_fer2013 --together
```

Necə işləyir:

- Hər konfiqurasiya **ayrıca prosesdə** işləyir, ona görə RAM ölçüsü təmiz olur.
- `--together` seçilmiş modelləri müqayisə rejimindəki kimi bir prosesdə, eyni kadrda işlədir. Bu,
  kompüterin onları birlikdə çəkib-çəkmədiyini göstərir.
- CPU → `onnx`; GPU → `torch` + CUDA (yalnız EmotiEffLib). CUDA yoxdursa və ya model quraşdırılmayıbsa,
  səbəbi göstərilərək keçilir.
- Kamera kadrları yalnız yaddaşda saxlanılır və ölçü bitən kimi silinir.
- Nəticə `reports/benchmark_<vaxt>.json` faylına yazılır (yalnız rəqəmlər).

Bu maşında ölçülüb (Windows 11, CPU, ONNX, 40 kadr, 512×512 şəkil):

| Model | Model ms (orta / p95) | Emal FPS | RAM + MB |
|---|---|---|---|
| enet_b0_8_best_afew | 12.5 / 15.3 | 35 | 71 |
| enet_b0_8_va_mtl | 12.2 / 13.9 | 36 | 71 |
| enet_b2_8 | 24.5 / 31.6 | 24 | 103 |
| mbf_va_mtl | 9.5 / 10.6 | 32 | 54 |
| deepface_emotion | 1.2 / 1.5 | 42 | 32 |
| vit_fer2013 | 199 / 216 | 4.5 | 357 |
| **enet_b0 + DeepFace + ViT birlikdə** | 299 / 321 | 3.1 | 412 |
| **6 model birlikdə** | 558 / 764 | 1.7 | 530 |

Detektor (YuNet) hər kadrda ~16–25 ms çəkir və bütün modellər üçün bir dəfə işləyir.

Nəticələrin şərhi:

- Saniyədə 1.5 analiz üçün interval 667 ms-dir.
- Default üçlük birlikdə ~300 ms-dir — sığır.
- 6 model birlikdə sığmır. "Avto" rejimi bu halda ən ağırlarını özü söndürür.
- Tək modeldə CPU yükü kiçikdir: UI ilə işləyərkən prosesin payı 4–10% olub.

### GPU

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements-gpu.txt
python scripts/download_models.py --torch
```

Sonra `config.yaml`-da `model.engine: torch` və `model.device: cuda` yazın. `torch.cuda.is_available()`
yoxlanılır; CUDA yoxdursa, CPU-ya keçilir və CSV-nin `device` sütununda bu görünür.

---

## Məxfilik

Qaydalar sözlə yox, **kodla** təmin olunur və testlərlə yoxlanılır.

| Qayda | Necə təmin olunur |
|---|---|
| Adi işdə video və şəkil diskə yazılmır | `cv2.imwrite`, `VideoWriter`, `PIL.Image.save` iş vaxtı bloklanır (`privacy.DiskImageGuard`). Mənbə kodda qadağan çağırışlar AST ilə yoxlanılır. Pipeline testi fayl sistemində yalnız `.csv` yarandığını təsdiqləyir. Önizləmə yalnız yaddaşda JPEG-dir. |
| Araşdırma istisnası — yalnız razılıqla | Şəkli yazan **yeganə** modul `src/dataset.py`-dır (AST testi başqa modullara icazə vermir). Yalnız nəzarətli testdə, tester həmin test üçün açarı yandırdıqda, yalnız "saxlama" kadrları, üz ətrafı kəsik, yalnız `research.dir` altına. Tester ID anonim olmalıdır. `research.save_test_frames: false` imkanı tam söndürür. Test zamanı da `cv2.imwrite` və s. bağlı qalır. |
| Şəbəkəyə heç nə göndərilmir | Modellər yükləndikdən sonra loopback-dən başqa bütün bağlantılar və DNS sorğuları bloklanır (`privacy.NetworkGuard`). Model çəkisi çatışmırsa, avtomatik endirmə əvəzinə xəta verilir. |
| UI şəbəkəsizdir | HTML/CSS/JS pəncərəyə inline yüklənir — **heç bir HTTP server və port açılmır**. CSP `default-src 'none'; connect-src 'none'`. WebView2 `private_mode`-da işləyir, SmartScreen/telemetriya/fon şəbəkəsi söndürülür. |
| Razılıq olmadan kamera açılmır | `CameraSource` yalnız razılıq düyməsinin yaratdığı `ConsentToken` ilə açılır; saxta token rədd edilir. |
| “Dayandır” | Panel və indikatordakı düymə kameranı dərhal bağlayır. Pəncərəni bağlamaq (Alt+F4 daxil) da kameranı dayandırır. |
| Kamera indikatoru | Kamera açıq olduğu müddətdə ekranın yuxarısında həmişə görünür |
| Ad yoxdur | Loglarda yalnız anonim `operator_id`. Ad kimi görünən ID config-də rədd edilir. |

Qeyd: WebView2 runtime-in özü Windows komponentidir və onun OS səviyyəsində diaqnostika siyasəti təşkilatın
Windows ayarlarından asılıdır. Proqramın Python prosesi və səhifə heç bir xarici bağlantı qurmur.

---

## Testlər

```bash
python -m pytest
```

~290 test. Real model çəkiləri olmadan modeldən asılı testlər `skip` olur (məsələn, Kaggle faylı gələnə qədər).

| Fayl | Nəyi yoxlayır |
|---|---|
| `test_detector.py` | Real YuNet: frontal üz, boş/küy kadr, ən böyük üzün seçilməsi, kəsik, kiçik, örtülmüş üz; landmark həndəsəsi |
| `test_recognizer.py` | 4 model, ehtimalların cəmi, RGB çevrilməsi, CUDA fallback, çəki çatışmazlığında endirmənin olmaması |
| `test_smoothing.py` | Tək kadrın göstərilməməsi, pəncərə sərhədi, 0.5 həddi, “üz yoxdur” |
| `test_csv_format.py` | Dəqiq sütun sırası, hər status üçün sətir, günlük fayl, sxem qoruması |
| `test_privacy.py` | Şəbəkə/DNS/UDP bloku, loopback icazəsi, `imwrite`/`VideoWriter`/PIL bloku, razılıq tokeni |
| `test_no_disk_writes.py` | **Video/şəkil diskə yazılmır:** AST yoxlaması + real pipeline-dan sonra fayl sistemi snapshot-ı |
| `test_pipeline.py`, `test_session.py` | `--video`, `--image`, canlı pipeline, razılıq → işləmə → dayandırma, sorğu cavabı və timeout, kor rejim |
| `test_ui.py` | Səhifələrin özünə-kifayətliyi, CSP, JS körpüsü |
| `test_model_zoo.py` | Model reyestri, FER etiket sırasının xəritəsi, FER modellərinin test şəklində nəticəsi |
| `test_keras_converter.py` | Keras .h5 → ONNX çeviricisi: sintetik CNN-lərdə Keras ilə 1e-5 dəqiqliklə eyni nəticə |
| `test_cameras.py` | Kamera siyahısı kameranı açmır, seçimin yadda qalması (cihaz yolu → ad → index), işləyərkən keçid, açılmayan kamerada köhnənin davamı, dayandırandan sonra heç bir kameranın açıq qalmaması, test zamanı keçidin bağlı olması |
| `test_variants.py` | Kəsik boşluğu, gözlərə görə düzləndirmə, flip TTA ortalaması, canlı detektorun eyni funksiyanı istifadəsi |
| `test_dataset.py` | Şəkil yalnız razılıqla və yalnız `research.dir`-ə; razılıqsız/config söndürülü halda heç nə; ad kimi tester ID-nin rəddi; oxuyucu və silmə CLI-ı |
| `test_evaluate.py` | Oflayn qiymətləndirici: metrikalar, işarə testi, LOTO, ayar təsiri, keş, CLI, hesabatda tester ID-lərinin olmaması |
| `test_calibration.py` | Nəzarətli test: zaman xətti, Neytral-birinci sıra, düzgünlük/balanslı hesab, yalnız saxlama kadrlarının `expected` ilə yazılması, test zamanı sorğunun açılmaması |
| `test_multimodel.py` | Eyni kadrda bir neçə model, hər model üçün ayrıca CSV sətri, büdcə, işləyərkən model keçidi, sorğuda hər modelin yazılması |
| `test_report.py` | Confusion matrix, kappa, Wilson, modellərin müqayisəsi və cüt-cüt razılaşma, operator ID-lərinin hesabatda olmaması |
| `test_benchmark.py` | Uçdan-uca benchmark (tək və birlikdə), GPU/FER keçidləri |

---

## Layihə strukturu

```
config.yaml              konfiqurasiya
requirements.txt         əsas asılılıqlar (CPU/ONNX)
requirements-gpu.txt     ixtiyari torch/CUDA
requirements-convert.txt yalnız bir dəfəlik model çevirməsi üçün (torch CPU, transformers, h5py, keras)
scripts/
  download_models.py     endirmə + ViT ixracı + Kaggle çevirməsi (--extra)
  keras_h5_to_onnx.py    TensorFlow-suz Keras .h5 → ONNX çevirici
src/
  app.py                 CLI giriş nöqtəsi (python -m src.app)
  config.py              config yüklənməsi və validasiyası
  privacy.py             şəbəkə/disk qoruyucuları, razılıq tokeni
  capture.py             webcam / video / şəkil mənbələri, işləyərkən kamera keçidi
  cameras.py             kameraların siyahısı (DirectShow, kameranı açmadan) və seçimin yadda saxlanması
  detector.py            YuNet + keyfiyyət filtri
  model_zoo.py           bütün modellərin reyestri və ümumi yaradıcı
  recognizer.py          EmotiEffLib qatı
  fer_models.py          FER2013 modelləri üçün ümumi ONNX adapteri
  smoothing.py           10 s hərəkətli ortalama
  csv_logger.py          CSV sxemi
  selfreport.py          sorğu planlayıcısı
  calibration.py         nəzarətli test: göstəriş planı, mərhələlər, düzgünlük hesabı
  variants.py            ön emal ayarları: kəsik boşluğu, düzləndirmə, flip TTA
  dataset.py             araşdırma dataseti (razılıqla şəkil yazan yeganə modul; python -m src.dataset)
  evaluate.py            oflayn model × ayar müqayisəsi (python -m src.evaluate)
  pipeline.py            kadr → üz → model(lər) → CSV, hesablama büdcəsi
  session.py             razılıq/işləmə/dayandırma həyat dövrü
  report.py              HTML hesabat (python -m src.report)
  benchmark.py           benchmark (python -m src.benchmark)
  ui/                    pywebview pəncərələri, Windows DWM köməkçiləri, web/ (HTML/CSS/JS)
tests/                   pytest
```

---

## Məhdudiyyətlər

- **Model üz ifadəsini ölçür, hissi yox.** Neytral üzlə əsəbi olmaq, yaxud nəzakət üçün gülümsəmək mümkündür.
- **Örtülmə aşkarlaması evristikdir.** Sintetik örtülmələr (rəngli blok, bulanıq dəri) tutulur. Real əl
  və barmaqlarda detal çox olduğu üçün bəzən keçə bilər. Pilotda `face_reason` statistikasını önizləmə ilə
  müqayisə edin; həddi `detector.min_region_detail` ilə dəyişin.
- **Baş əyilməsi.** YuNet landmark-ları baş əyiləndə bucağı az göstərir, ona görə `tilted` yoxlaması yalnız
  kəskin hallarda işə düşür.
- **“Yorğun” üçün model sinfi yoxdur.** `_mtl` modellərində ixtiyari VA qaydası ilə sınaqdan keçirmək olar.
- **Statistik etibarlılıq.** Etibarlı nəticə üçün ən azı 100 müqayisə oluna bilən cavab və ≥ 3 operator tövsiyə olunur.
- **FER2013 modelləri** 48×48 boz, aşağı keyfiyyətli şəkillərdə öyrədilib. Webcam görüntüsündə davranışları
  fərqli ola bilər; "Həqarət" sinfini heç vaxt vermirlər. ViT-in iddia olunan 71.5% dəqiqliyi FER2013 test
  dəsti üzrədir, sizin şəraitiniz üzrə deyil — müqayisə rejimi məhz bunu yoxlamaq üçündür.
- **Modellər arası razılaşma** düzgünlük demək deyil: eyni datasetdə öyrədilmiş modellər eyni səhvi edə bilər.
- Eynək, qulaqlıq mikrofonu, zəif işıq və kamera bucağı nəticəyə təsir edir. Bunları pilotun başında yoxlayın.

---

## Qeyd: lisenziya və istifadə

- Kod və EmotiEffLib kitabxanası **Apache-2.0** lisenziyalıdır.
- Model çəkiləri **AffectNet** dataseti üzərində öyrədilib. AffectNet **qeyri-kommersiya akademik lisenziya** ilə
  paylanır. **Kommersiya istifadəsindən əvvəl hüquqi yoxlama lazımdır.**
- YuNet modeli (OpenCV Zoo) MIT lisenziyalıdır.
- **FER2013 modelləri:**
  - DeepFace kitabxanası MIT-dir, Hugging Face ViT modeli Apache-2.0, Kaggle modeli CC0 kimi paylanır.
  - Hamısı **FER2013** datasetində öyrədilib. FER2013-ün istifadə şərtləri (Kaggle müsabiqə məlumatı,
    tədqiqat məqsədli) ayrıca aydınlaşdırılmalıdır — kommersiya istifadəsindən əvvəl bunu da hüquqi yoxlamaya daxil edin.
  - `manish-9245/Facial-Emotion-Recognition-using-OpenCV-and-Deepface` reposunun lisenziyası **yalnız
    qeyri-kommersiya** istifadəyə icazə verir. Bu layihə həmin reponun kodunu və Haar cascade-ini istifadə
    etmir — yalnız DeepFace-in (MIT) emosiya modelini birbaşa qoşur.
- Model **daxili hissi yox, üz ifadəsini** ölçür. Nəticələr fərdi qiymətləndirmə, mükafatlandırma və ya
  intizam qərarları üçün istifadə edilməməlidir.
