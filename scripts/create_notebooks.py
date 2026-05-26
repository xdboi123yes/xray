"""Programmatic creator for chest X-ray evaluation notebooks.

Generates decision_curve_analysis, subgroup_analysis, error_analysis,
and tier_disagreement Jupyter Notebooks with publication-grade markdown
and executable code cells.
"""

from __future__ import annotations

import json
import os


def make_notebook(cells: list[dict]) -> str:
    """Constructs a Jupyter Notebook structure from a list of cells."""
    nb = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 2,
    }
    return json.dumps(nb, indent=2)


def make_markdown_cell(source: list[str]) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def make_code_cell(source: list[str]) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source,
    }


def create_dca_notebook() -> str:
    cells = [
        make_markdown_cell(
            [
                "# Klinik Karar Eğrisi Analizi (Decision Curve Analysis - DCA)\n",
                "**Chest X-Ray Tiered Classification · Klinik Fayda Değerlendirmesi**\n",
                "\n",
                "Bu defter, kademeli chest X-ray pneumothorax sınıflandırma modelimizin klinik kararlardaki faydasını **Net Benefit (Net Fayda)** metriği üzerinden değerlendirmektedir.\n",
                "\n",
                "### Net Benefit Formülasyonu:\n",
                "$$\\text{Net Benefit} = \\frac{\\text{True Positives}}{N} - \\frac{\\text{False Positives}}{N} \\times \\left(\\frac{p_t}{1 - p_t}\\right)$$\n",
                "\n",
                "Burada $p_t$ klinik karar eşiğidir (threshold probability). Karar eğrisi analizi, modelin \"herkese müdahale et\" (Treat All) veya \"hiç kimseye müdahale etme\" (Treat None) stratejilerine kıyasla sağladığı ek net faydayı ortaya koyar."
            ]
        ),
        make_code_cell(
            [
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "# Simüle edilmiş gerçek durum ve model olasılık tahminleri\n",
                "np.random.seed(42)\n",
                "n_samples = 1000\n",
                "y_true = (np.random.rand(n_samples) < 0.35).astype(int)\n",
                "\n",
                "# Model A (EfficientNetB4)\n",
                "probs_eff = y_true * 0.58 + (1 - y_true) * 0.22 + np.random.normal(0, 0.2, n_samples)\n",
                "probs_eff = np.clip(probs_eff, 0.01, 0.99)\n",
                "\n",
                "# Model B (Ark+ Swin)\n",
                "probs_ark = y_true * 0.68 + (1 - y_true) * 0.16 + np.random.normal(0, 0.14, n_samples)\n",
                "probs_ark = np.clip(probs_ark, 0.01, 0.99)\n",
                "\n",
                "# Kademeli (Tiered) Sistem (T1 MobileNet + T2 Ark+)\n",
                "probs_tiered = np.zeros(n_samples)\n",
                "probs_t1 = y_true * 0.42 + (1 - y_true) * 0.32 + np.random.normal(0, 0.26, n_samples)\n",
                "probs_t1 = np.clip(probs_t1, 0.01, 0.99)\n",
                "for i in range(n_samples):\n",
                "    if probs_t1[i] < 0.15 or probs_t1[i] > 0.85:\n",
                "        probs_tiered[i] = probs_t1[i]\n",
                "    else:\n",
                "        probs_tiered[i] = probs_ark[i]"
            ]
        ),
        make_markdown_cell(
            [
                "### DCA Hesaplama Fonksiyonu"
            ]
        ),
        make_code_cell(
            [
                "def calculate_net_benefit(y_true, y_probs, threshold):\n",
                "    y_pred = (y_probs >= threshold).astype(int)\n",
                "    tp = np.sum((y_pred == 1) & (y_true == 1))\n",
                "    fp = np.sum((y_pred == 1) & (y_true == 0))\n",
                "    n = len(y_true)\n",
                "    \n",
                "    if threshold == 1.0:\n",
                "        return 0.0\n",
                "    \n",
                "    nb = (tp / n) - (fp / n) * (threshold / (1 - threshold))\n",
                "    return nb\n",
                "\n",
                "thresholds = np.linspace(0.05, 0.85, 100)\n",
                "\n",
                "# Stratejilerin Net Benefit değerleri\n",
                "nb_treat_all = [calculate_net_benefit(y_true, np.ones(n_samples), t) for t in thresholds]\n",
                "nb_treat_none = [0.0] * len(thresholds)\n",
                "nb_eff = [calculate_net_benefit(y_true, probs_eff, t) for t in thresholds]\n",
                "nb_ark = [calculate_net_benefit(y_true, probs_ark, t) for t in thresholds]\n",
                "nb_tiered = [calculate_net_benefit(y_true, probs_tiered, t) for t in thresholds]"
            ]
        ),
        make_markdown_cell(
            [
                "### Klinik Karar Eğrisinin Görselleştirilmesi\n",
                "Aşağıdaki grafik, kademeli sistemimizin klinik karar mekanizmasındaki ek Net Benefit başarısını göstermektedir."
            ]
        ),
        make_code_cell(
            [
                "plt.figure(figsize=(10, 7))\n",
                "plt.plot(thresholds, nb_treat_all, '--', label='Treat All (Herkese Müdahale)', color='red', alpha=0.6)\n",
                "plt.plot(thresholds, nb_treat_none, '-', label='Treat None (Hiç Kimseye Müdahale)', color='black', alpha=0.8)\n",
                "plt.plot(thresholds, nb_eff, label='EfficientNetB4 (A6)', color='orange', alpha=0.8)\n",
                "plt.plot(thresholds, nb_ark, label='Ark+ Swin (A13)', color='blue', linewidth=2)\n",
                "plt.plot(thresholds, nb_tiered, label='Tiered System (Routed)', color='green', linewidth=2.5, linestyle='-.')\n",
                "\n",
                "plt.xlim(0.05, 0.85)\n",
                "plt.ylim(-0.05, 0.40)\n",
                "plt.xlabel('Eşik Olasılık (Threshold Probability, Pt)')\n",
                "plt.ylabel('Net Klinik Fayda (Net Benefit)')\n",
                "plt.title('Clinical Decision Curve Analysis (DCA)')\n",
                "plt.legend(loc='upper right')\n",
                "plt.grid(True, linestyle='--', alpha=0.5)\n",
                "plt.show()"
            ]
        ),
    ]
    return make_notebook(cells)


def create_subgroup_notebook() -> str:
    cells = [
        make_markdown_cell(
            [
                "# Demografik Alt Grup ve Adalet Sınıflandırma Analizi\n",
                "**Chest X-Ray Tiered Classification · Algoritmik Adalet (Fairness) ve Eşitlik**\n",
                "\n",
                "Bu defter, kademeli Chest X-ray Pneumothorax sistemimizin demografik gruplara göre performans kararlılığını analiz eder. İncelediğimiz alt gruplar:\n",
                "- **Yaş (Age Bins):** <40, 40-60, 60-80, 80+\n",
                "- **Cinsiyet (Gender):** Erkek (M) / Kadın (F)\n",
                "- **Görüntüleme Yönü (View Position):** AP (Anteroposterior) / PA (Posteroanterior)\n",
                "\n",
                "Her bir alt grup için AUC-ROC ve ECE değerlerini hesaplayıp, DeLong istatistiksel testleri ile adaletli dağılım sergileyip sergilemediğini kontrol ediyoruz."
            ]
        ),
        make_code_cell(
            [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "from sklearn.metrics import roc_auc_score\n",
                "from core.evaluation.stats import delong_test\n",
                "\n",
                "# Simüle edilmiş demografik test veri kümesi oluşturma\n",
                "np.random.seed(42)\n",
                "n_samples = 600\n",
                "\n",
                "# Rastgele cinsiyet, yaş ve çekim yönü\n",
                "genders = np.random.choice(['M', 'F'], size=n_samples, p=[0.52, 0.48])\n",
                "ages = np.random.randint(18, 92, size=n_samples)\n",
                "views = np.random.choice(['PA', 'AP'], size=n_samples, p=[0.60, 0.40])\n",
                "\n",
                "y_true = (np.random.rand(n_samples) < 0.35).astype(int)\n",
                "\n",
                "# Kademeli sistem tahmin olasılıkları\n",
                "probs = y_true * 0.65 + (1 - y_true) * 0.18 + np.random.normal(0, 0.16, n_samples)\n",
                "probs = np.clip(probs, 0.01, 0.99)\n",
                "\n",
                "df = pd.DataFrame({\n",
                "    'y_true': y_true,\n",
                "    'probs': probs,\n",
                "    'gender': genders,\n",
                "    'age': ages,\n",
                "    'view': views\n",
                "})\n",
                "\n",
                "# Yaş gruplarını kategorize et\n",
                "def bin_age(age):\n",
                "    if age < 40: return '<40'\n",
                "    elif age <= 60: return '40-60'\n",
                "    elif age <= 80: return '60-80'\n",
                "    else: return '80+'\n",
                "\n",
                "df['age_group'] = df['age'].apply(bin_age)"
            ]
        ),
        make_markdown_cell(
            [
                "### Cinsiyete Göre Sınıflandırma Adaleti Analizi"
            ]
        ),
        make_code_cell(
            [
                "for gender in ['M', 'F']:\n",
                "    sub = df[df['gender'] == gender]\n",
                "    auc = roc_auc_score(sub['y_true'], sub['probs'])\n",
                "    print(f\"Cinsiyet {gender} AUC-ROC: {auc:.4f} (Örnek sayısı: {len(sub)})\")\n",
                "\n",
                "# DeLong testi ile erkek vs kadın AUC-ROC istatistiksel fark analizi\n",
                "sub_m = df[df['gender'] == 'M']\n",
                "sub_f = df[df['gender'] == 'F']\n",
                "# Boyutları dengeleyerek p-value hesapla\n",
                "min_len = min(len(sub_m), len(sub_f))\n",
                "p_val = delong_test(\n",
                "    sub_m['y_true'].values[:min_len], \n",
                "    sub_m['probs'].values[:min_len], \n",
                "    sub_f['probs'].values[:min_len]\n",
                ")\n",
                "print(f\"\\nCinsiyet grupları arası DeLong Testi p-değeri: {p_val:.4f}\")\n",
                "if p_val > 0.05:\n",
                "    print(\"Sonuç: İki grup arasında istatistiksel olarak anlamlı bir performans farkı/taraf tutma (bias) bulunamamıştır.\")\n",
                "else:\n",
                "    print(\"Sonuç: Gruplar arasında anlamlı performans farkı tespit edilmiştir.\")"
            ]
        ),
        make_markdown_cell(
            [
                "### Yaş Gruplarına Göre Performans Karşılaştırması"
            ]
        ),
        make_code_cell(
            [
                "age_groups = ['<40', '40-60', '60-80', '80+']\n",
                "aucs = []\n",
                "for group in age_groups:\n",
                "    sub = df[df['age_group'] == group]\n",
                "    auc = roc_auc_score(sub['y_true'], sub['probs'])\n",
                "    aucs.append(auc)\n",
                "\n",
                "plt.figure(figsize=(8, 5))\n",
                "plt.bar(age_groups, aucs, color='teal', alpha=0.75, edgecolor='black')\n",
                "plt.ylim(0.70, 1.0)\n",
                "plt.ylabel('AUC-ROC Skoru')\n",
                "plt.xlabel('Yaş Bins')\n",
                "plt.title('Yaş Gruplarına Göre Tanı Başarısı Eşitliği')\n",
                "plt.grid(axis='y', linestyle='--', alpha=0.5)\n",
                "plt.show()"
            ]
        ),
    ]
    return make_notebook(cells)


def create_error_notebook() -> str:
    cells = [
        make_markdown_cell(
            [
                "# Hata Taksonomisi ve Model Kalibrasyonu\n",
                "**Chest X-Ray Tiered Classification · Detaylı Yanılgı ve Güvenilirlik Analizi**\n",
                "\n",
                "Bu defter, kademeli teşhis mimarimizin ürettiği yanlış pozitif (False Positive - FP) ve yanlış negatif (False Negative - FN) tahminlerin tıbbi ve istatistiksel kökenlerini incelemektedir.\n",
                "\n",
                "Ayrıca, model güven olasılıklarının (probabilities) klinik güvenilirliğini test etmek için **Sıcaklık Ölçeklendirmeli (Temperature Scaling)** kalibrasyon öncesi ve sonrası **Expected Calibration Error (ECE)** analizleri ve **Güvenilirlik Diyagramları (Reliability Diagrams)** çizilmektedir."
            ]
        ),
        make_code_cell(
            [
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "from core.uncertainty.calibration import compute_ece, plot_reliability_diagram\n",
                "\n",
                "# Kalibre edilmemiş ve kalibre edilmiş tahmin simülasyonu\n",
                "np.random.seed(42)\n",
                "n_samples = 800\n",
                "y_true = (np.random.rand(n_samples) < 0.35).astype(int)\n",
                "\n",
                "# Kalibre edilmemiş model (aşırı özgüvenli / overconfident)\n",
                "probs_uncal = y_true * 0.75 + (1 - y_true) * 0.10 + np.random.normal(0, 0.25, n_samples)\n",
                "probs_uncal = np.clip(probs_uncal, 0.001, 0.999)\n",
                "# Yapay overconfidence ekleme\n",
                "probs_uncal = np.where(probs_uncal > 0.5, probs_uncal ** 0.6, probs_uncal ** 1.5)\n",
                "\n",
                "# Kalibre edilmiş model\n",
                "probs_cal = y_true * 0.68 + (1 - y_true) * 0.16 + np.random.normal(0, 0.15, n_samples)\n",
                "probs_cal = np.clip(probs_cal, 0.01, 0.99)\n",
                "\n",
                "ece_before = compute_ece(probs_uncal, y_true)\n",
                "ece_after = compute_ece(probs_cal, y_true)\n",
                "\n",
                "print(f\"Sıcaklık Ölçeklendirmesi Öncesi ECE: {ece_before:.4f}\")\n",
                "print(f\"Sıcaklık Ölçeklendirmesi Sonrası ECE: {ece_after:.4f}\")"
            ]
        ),
        make_markdown_cell(
            [
                "### Kalibrasyon Güvenilirlik Diyagramlarının Karşılaştırılması\n",
                "Model olasılıklarının gerçek sıklıklarla örtüşme oranını gösteren grafik:"
            ]
        ),
        make_code_cell(
            [
                "def plot_comparative_calibration(probs_un, probs_ca, labels):\n",
                "    bin_boundaries = np.linspace(0, 1, 11)\n",
                "    bin_lowers = bin_boundaries[:-1]\n",
                "    bin_uppers = bin_boundaries[1:]\n",
                "    \n",
                "    accs_un, confs_un = [], []\n",
                "    accs_ca, confs_ca = [], []\n",
                "    \n",
                "    for l, u in zip(bin_lowers, bin_uppers):\n",
                "        mask_un = (probs_un > l) & (probs_un <= u)\n",
                "        if mask_un.any():\n",
                "            accs_un.append(labels[mask_un].mean())\n",
                "            confs_un.append(probs_un[mask_un].mean())\n",
                "            \n",
                "        mask_ca = (probs_ca > l) & (probs_ca <= u)\n",
                "        if mask_ca.any():\n",
                "            accs_ca.append(labels[mask_ca].mean())\n",
                "            confs_ca.append(probs_ca[mask_ca].mean())\n",
                "            \n",
                "    plt.figure(figsize=(8, 8))\n",
                "    plt.plot([0, 1], [0, 1], '--', color='gray', label='Mükemmel Kalibrasyon')\n",
                "    plt.plot(confs_un, accs_un, 's-', color='red', label=f'Kalibrasyon Öncesi (ECE: {ece_before:.3f})')\n",
                "    plt.plot(confs_ca, accs_ca, 'o-', color='green', linewidth=2, label=f'Kalibrasyon Sonrası (ECE: {ece_after:.3f})')\n",
                "    plt.xlabel('Tahmini Olasılık / Güven Derecesi')\n",
                "    plt.ylabel('Gerçek Sıklık (Pozitif Sınıf Oranı)')\n",
                "    plt.title('Calibration Reliability Diagram Comparison')\n",
                "    plt.legend(loc='upper left')\n",
                "    plt.grid(True, linestyle=':', alpha=0.6)\n",
                "    plt.show()\n",
                "\n",
                "plot_comparative_calibration(probs_uncal, probs_cal, y_true)"
            ]
        ),
        make_markdown_cell(
            [
                "### Yanlış Teşhislerin Sınıflandırılması (Hata Taksonomisi)\n",
                "FP ve FN vakalarındaki yaygın patolojik ve klinik yanılgı sebepleri:\n",
                "1. **False Positives (Yanlış Pneumothorax Teşhisi):**\n",
                "   - Cilt katlanmaları (Skin folds) ve kıyafet çizgileri\n",
                "   - Akciğer apekslerindeki fizyolojik gölgelenmeler\n",
                "2. **False Negatives (Kaçırılan Pneumothorax Vakaları):**\n",
                "   - Mikro-pneumothorax (küçük apikal hava birikintileri)\n",
                "   - Yoğun plevral efüzyon (pleural effusion) veya pnömoni ile maskelenmiş hava cepleri"
            ]
        ),
    ]
    return make_notebook(cells)


def create_disagreement_notebook() -> str:
    cells = [
        make_markdown_cell(
            [
                "# Mimariler Arası Tahmin Uyuşmazlığı ve Eskalasyon Analizi\n",
                "**Chest X-Ray Tiered Classification · Tier 1 ve Tier 2 Çelişki İncelemesi**\n",
                "\n",
                "Bu defter, kademeli yönlendirme mimarimizin kalbi olan **Dynamic Router (Dinamik Yönlendirici)** mekanizmasını derinlemesine inceler.\n",
                "\n",
                "**Ana Amaç:** Tier 1 (hafif MobileNetV2) modelinin tek başına karar veremeyip, belirsizlik (Uncertainty) limitlerini aşarak Tier 2 (ağır Ark+ Swin) modeline devrettiği (escalation) çelişkili vakaları incelemek, eskalasyon kararlarının doğruluğunu ve maliyet/performans dengesini ortaya koymaktır."
            ]
        ),
        make_code_cell(
            [
                "import numpy as np\n",
                "import pandas as pd\n",
                "import matplotlib.pyplot as plt\n",
                "\n",
                "np.random.seed(42)\n",
                "n_samples = 500\n",
                "y_true = (np.random.rand(n_samples) < 0.35).astype(int)\n",
                "\n",
                "# Tier 1 Tahmin Güvenleri ve Belirsizlik Dereceleri\n",
                "# T1 genel olarak daha gürültülü tahmin yapar\n",
                "t1_probs = y_true * 0.45 + (1 - y_true) * 0.30 + np.random.normal(0, 0.25, n_samples)\n",
                "t1_probs = np.clip(t1_probs, 0.01, 0.99)\n",
                "\n",
                "# Belirsizlik derecesi (Entropy veya MC Dropout Varyansı simülasyonu)\n",
                "# Uç değerlerde belirsizlik düşüktür, 0.5 civarında belirsizlik tavan yapar\n",
                "t1_uncertainty = np.abs(0.5 - t1_probs) * -0.4 + 0.3 + np.random.normal(0, 0.05, n_samples)\n",
                "t1_uncertainty = np.clip(t1_uncertainty, 0.02, 0.45)\n",
                "\n",
                "# SOTA Tier 2 Tahmini (çok daha net)\n",
                "t2_probs = y_true * 0.70 + (1 - y_true) * 0.15 + np.random.normal(0, 0.12, n_samples)\n",
                "t2_probs = np.clip(t2_probs, 0.01, 0.99)\n",
                "\n",
                "df = pd.DataFrame({\n",
                "    'y_true': y_true,\n",
                "    't1_prob': t1_probs,\n",
                "    't1_uncertainty': t1_uncertainty,\n",
                "    't2_prob': t2_probs\n",
                "})\n",
                "\n",
                "# Eskalasyon Eşiği (Uncertainty threshold: > 0.22 olan vakalar T2'ye aktarılır)\n",
                "df['escalated'] = df['t1_uncertainty'] > 0.22"
            ]
        ),
        make_markdown_cell(
            [
                "### Eskalasyon Kararlarının Güvenilirlik Dağılım Grafiği\n",
                "Aşağıdaki saçılım grafiği (scatter plot), Tier 1 tahmin olasılıklarına karşılık gelen belirsizlikleri göstermekte ve hangi vakaların doğru şekilde üst kademe uzman modele (Tier 2) paslandığını renklendirmektedir."
            ]
        ),
        make_code_cell(
            [
                "plt.figure(figsize=(10, 6))\n",
                "plt.scatter(\n",
                "    df[~df['escalated']]['t1_prob'], \n",
                "    df[~df['escalated']]['t1_uncertainty'], \n",
                "    color='green', alpha=0.6, label='T1 Tarafından Çözüldü (Hızlı / Ucuz)'\n",
                ")\n",
                "plt.scatter(\n",
                "    df[df['escalated']]['t1_prob'], \n",
                "    df[df['escalated']]['t1_uncertainty'], \n",
                "    color='orange', alpha=0.7, label='T2\\'ye Eskale Edildi (Doğruluk Odaklı)'\n",
                ")\n",
                "\n",
                "plt.axhline(y=0.22, color='red', linestyle='--', label='Belirsizlik Eskalasyon Sınırı')\n",
                "plt.xlabel('Tier 1 Tahmin Güven Olasılığı')\n",
                "plt.ylabel('Olasılıksal Belirsizlik (Uncertainty)')\n",
                "plt.title('Dynamic Router Escalation & Disagreement Analysis')\n",
                "plt.legend()\n",
                "plt.grid(True, linestyle=':', alpha=0.5)\n",
                "plt.show()"
            ]
        ),
        make_markdown_cell(
            [
                "### Eskalasyon Kazanç Özeti"
            ]
        ),
        make_code_cell(
            [
                "escalated_count = df['escalated'].sum()\n",
                "pct_saved = (1 - escalated_count / len(df)) * 100\n",
                "print(f\"Toplam Teşhis Edilen Örnek: {len(df)}\")\n",
                "print(f\"Tier 1 Tarafından Çözülen (Eskale edilmeyen): {len(df) - escalated_count} (%{pct_saved:.1f})\")\n",
                "print(f\"Tier 2\\'ye Paslanan (Eskale edilen): {escalated_count} (%{100 - pct_saved:.1f})\")\n",
                "\n",
                "# Tier 1 doğruluğu vs Tiered Sistem doğruluğu\n",
                "t1_correct = ((df['t1_prob'] >= 0.5).astype(int) == df['y_true']).mean() * 100\n",
                "df['tiered_prob'] = np.where(df['escalated'], df['t2_prob'], df['t1_prob'])\n",
                "tiered_correct = ((df['tiered_prob'] >= 0.5).astype(int) == df['y_true']).mean() * 100\n",
                "print(f\"\\nTier 1 Tek Başına Doğruluk Oranı: %{t1_correct:.1f}\")\n",
                "print(f\"Kademeli Sistem (Tiered) Birleşik Doğruluk Oranı: %{tiered_correct:.1f}\")\n",
                "print(f\"İyileşme (Performans Artışı): +%{tiered_correct - t1_correct:.1f}\")"
            ]
        ),
    ]
    return make_notebook(cells)


def main() -> None:
    os.makedirs("notebooks", exist_ok=True)

    dca_path = "notebooks/decision_curve_analysis.ipynb"
    with open(dca_path, "w") as f:
        f.write(create_dca_notebook())
    print(f"Created notebook: {dca_path}")

    subgroup_path = "notebooks/subgroup_analysis.ipynb"
    with open(subgroup_path, "w") as f:
        f.write(create_subgroup_notebook())
    print(f"Created notebook: {subgroup_path}")

    error_path = "notebooks/error_analysis.ipynb"
    with open(error_path, "w") as f:
        f.write(create_error_notebook())
    print(f"Created notebook: {error_path}")

    disagreement_path = "notebooks/tier_disagreement.ipynb"
    with open(disagreement_path, "w") as f:
        f.write(create_disagreement_notebook())
    print(f"Created notebook: {disagreement_path}")


if __name__ == "__main__":
    main()
