"""İnterfeys və hesabat üçün Azərbaycan dilində adlar."""

EMOTION_LABELS = {
    "Anger": "Qəzəb",
    "Contempt": "Həqarət",
    "Disgust": "İkrah",
    "Fear": "Qorxu",
    "Happiness": "Xoşbəxtlik",
    "Neutral": "Neytral",
    "Sadness": "Kədər",
    "Surprise": "Təəccüb",
}

# Nəzarətli test göstərişləri: necə göstərmək (FACS-a yaxın, sadə dillə)
EXPRESSION_HINTS = {
    "Neutral": "Sakit, ifadəsiz baxın — üz əzələləri rahat",
    "Happiness": "Gülümsəyin — yanaqlar qalxsın, dişlər görünə bilər",
    "Sadness": "Kədərli görünün — dodaq kənarları aşağı, qaşların içi yuxarı",
    "Surprise": "Təəccüblənin — qaşlar yuxarı, gözlər geniş, ağız açıq",
    "Anger": "Qəzəbli görünün — qaşlar çatılıb aşağı, dodaqlar sıxılıb",
    "Fear": "Qorxmuş görünün — gözlər geniş açıq, qaşlar yuxarı və bir-birinə yaxın",
    "Disgust": "İkrah — burnu qırışdırın, üst dodaq yuxarı",
    "Contempt": "Həqarət — dodağın yalnız bir kənarı yuxarı (yarımçıq təbəssüm)",
}

STATUS_LABELS = {
    "ok": "Müəyyən",
    "uncertain": "Qeyri-müəyyən",
    "no_face": "Üz yoxdur",
    "warming_up": "Toplanır…",
}

# Geyim yoxlaması (python -m src.attire): nəticələr və config-dəki cavab adları.
# Config-ə burada olmayan cavab əlavə olunsa, adın özü göstərilir.
ATTIRE_LABELS = {
    "formal": "Rəsmi",
    "not_formal": "Qeyri-rəsmi",
    "unknown": "Qiymətləndirilə bilmir",
    "shirt": "düyməli köynək",
    "tshirt": "futbolka",
    "polo": "polo",
    "knitwear": "sviter / hudi",
    "other": "digər geyim",
    "white": "ağ",
    "light_blue": "açıq mavi",
    "beige": "bej / krem",
    "grey": "boz",
    "dark": "tünd",
    "pink": "çəhrayı",
    "patterned": "naxışlı",
    "bright": "parlaq rəngli",
}

FACE_REASON_LABELS = {
    "ok": "Üz aşkarlandı",
    "not_found": "Üz tapılmadı",
    "low_confidence": "Üz aydın görünmür",
    "partial": "Üz kadrdan çıxıb",
    "too_small": "Kameradan çox uzaqsınız",
    "profile": "Üz yana çevrilib",
    "tilted": "Baş çox əyilib",
    "occluded": "Üz örtülüb (əl, mikrofon…)",
    "camera_dark": "Kamera görüntüsü qaradır — qapaq və ya kamera düyməsi bağlı ola bilər",
}
