# Pokerbot_performance_analysis
# 🃏 IIT Pokerbots — Analyzer Dashboard

A professional log analysis framework + Streamlit web dashboard for **Sneak Peek Hold'em** bots.

---

## 🚀 Quick Start

```bash
pip install -r requirements.txt
streamlit run app.py
```
Open **http://localhost:8501** in your browser, then drag your `.glog` files into the sidebar.

---

## 📂 Project Structure

```
pokerbot_analyzer/
├── app.py                  ← 🖥  Streamlit dashboard (start here)
├── parser.py               ← Log parser → DataFrames
├── metrics.py              ← Strategic metrics engine
├── leak_detection.py       ← Automated leak detector
├── visualization.py        ← Matplotlib charts (CLI use)
├── comparison.py           ← Version comparison engine
├── report.py               ← HTML + text report generator
├── main.py                 ← CLI entry point
├── generate_sample_log.py  ← Test data generator
└── requirements.txt
```

---

## 🖥 Dashboard Tabs

| Tab | What's Inside |
|-----|---------------|
| **📊 Overview** | KPI bar, bankroll curve, win rate, position breakdown, preflop metrics, action frequency |
| **🎯 Auction** | Bid distribution, auction EV by outcome, bid vs hand strength, bid timeline |
| **🌊 Post-Flop** | Fold heatmap, barrel EV, c-bet/barrel metrics, action sunburst |
| **🔍 Hand Browser** | Filter & inspect individual hands by result / street / auction |
| **🚨 Leaks** | Severity donut, category breakdown, per-leak recommendations, download report |
| **🔄 Compare** | Upload 2+ logs → delta table + bar chart + overlaid bankroll curves |
| **📋 Data Explorer** | Browse raw DataFrames, filter columns, export CSV/JSON |

---

## 💻 CLI Usage (no UI needed)

```bash
python main.py --log game.glog --bot phoenix_1
python main.py --logs game1.glog game2.glog --bot phoenix_1
python main.py --compare v6:game_v6.glog v8:game_v8.glog --bot phoenix_1
```

---

## 🔮 Leak Detection (15+ rules)

Covers: preflop tightness/looseness, VPIP-PFR gap, fold-to-raise, auction overbidding, close losses, negative EV when winning auction, flat bidding vs hand strength, c-bet extremes, triple barrel frequency, river over-folding, single-round loss rate, high variance.

Each leak includes severity (🔴/🟡/🟢), evidence, and a concrete fix recommendation.
