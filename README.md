# Predikce finančních trhů pomocí Transformerů (TFT)

Tento projekt se zabývá **predikcí vývoje na finančních trzích** pomocí **transformerových modelů**, konkrétně **decoder-only architektury TFT (Temporal Fusion Transformer)**.  
Cílem bylo ověřit, jak dobře lze tímto přístupem modelovat časové řady akcií, indexů a kryptoměn, pokud jsou doplněny o **behaviorální faktory** jako je **investorova pozornost**, **sentiment** a **aktivita na sociálních sítích**.

---

## Cíl projektu

- Predikovat budoucí vývoj finančních časových řad pomocí moderních transformerových modelů.  
- Zahrnout do predikce **investorovu pozornost** získanou z textových dat (sociální sítě, články, diskuzní fóra).  
- Porovnat výsledky s tradičními přístupy (ARIMA, LSTM, GAN).  

---

## Struktura projektu

Projekt zahrnuje:

1. **Shromažďování a fúzi dat**
   - Historické ceny akcií, indexů a kryptoměn  
   - Textová data z Redditu a finančních zpravodajských webů
   - Data vyhledávání z Google Trends  

2. **Analýzu sentimentu a pozornosti investorů**
   - Pomocí modelu **FinBERT**  
   - Výpočet denního sentimentu a jeho agregace

3. **Modelování pomocí TFT (Transformer Time-series Forecasting)**
   - Architektura kombinuje **attention mechanismus** a **rekurentní složky** pro efektivní práci s časovými závislostmi 

4. **Vyhodnocení výkonu**
   - Různé metriky pro trendy a celkovou chybu
   - Vizualizace predikcí vs. reálných hodnot  

---

## Použité technologie

- **Python** – NumPy, Pandas, PyTorch, Scikit-learn  
- **Hugging Face Transformers**
- **FinBERT** – analýza sentimentu  
- **Plotly** – vizualizace  
- **Jupyter Notebook** – experimentální prostředí  

---

## Ukázka výsledku

Níže je příklad predikce pro akcie **Apple (AAPL)**, kde model zohledňuje i míru pozornosti investorů (aktivita na Twitteru a sentiment v médiích):

| Graf | Popis |
|------|--------|
| ![Výsledek 1](./text/apple.png) | Predikce ceny akcií Apple pomocí TTF modelu s integrací investorovy pozornosti a sentimentu z textových dat. |

---

## Dokumentace

Detailní rozbor metodologie, architektury modelu a výsledků naleznete v přiložené práci:

[**Text práce (PDF)**](./graphs_text/diplomova_prace_jezek.pdf)

---
