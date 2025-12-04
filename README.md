# Reinforcement-Learning-Trading-Bot-RLTrader-
Entwicklung eines autonomen Trading-Bots, der mit Reinforcement Learning (RL) lernt, profitable Handelsentscheidungen auf Basis historischer und aktueller Marktdaten zu treffen.

---
## Theorie RL
### RL erklärt: 
Ein Agent ist wie ein kleines Kind, das etwas Neues lernen will. Die Umgebung ist der Spielplatz, auf dem es spielt. Das Kind probiert Dinge aus, zum Beispiel klettern, rutschen oder schaukeln.
Für gute Aktionen bekommt es Belohnungen (wie ein Lob oder ein Gummibärchen). Für schlechte Aktionen bekommt es keine Belohnung oder vielleicht eine kleine Warnung.
Mit der Zeit merkt das Kind:
„Wenn ich bestimmte Dinge tue, bekomme ich mehr Belohnungen!“
Also versucht es immer klüger zu werden und herauszufinden, welche Entscheidungen ihm langfristig am meisten Gutes bringen.
Genau das ist Reinforcement Learning: Ausprobieren, Belohnung bekommen, und Schritt für Schritt besser werden.
#### Trading-Kontext
Ein Trading-Agent ist wie ein kleines Kind, das Börse spielen lernt. Die Umgebung ist nicht der Spielplatz, sondern der Markt also die Kurse, die rauf und runter gehen.
Der Agent probiert verschiedene Dinge aus so wie ein Kind:
Er kauft manchmal eine Aktie, er verkauft sie oder er wartet einfach.
Wenn der Agent durch seine Aktion Geld verdient, bekommt er eine Belohnung (wie ein Gummibärchen). Wenn er Geld verliert, bekommt er eine schlechte Rückmeldung (wie ein kleines „Nein, das war nicht gut“).
Mit der Zeit merkt der Agent: „Wenn ich in bestimmten Situationen kaufe oder verkaufe, bekomme ich mehr Belohnung!“
Darum versucht er immer besser zu verstehen, welche Entscheidungen ihm auf Dauer am meisten Gewinn bringen, genau wie ein Kind lernt, welche Spiele am meisten Spaß machen.
So funktioniert Reinforcement Learning im Trading: Der Agent probiert aus, lernt aus seinen Gewinnen und Fehlern und wird Schritt für Schritt ein besserer Trader.

---

## Environment
Wie ist ein Enviroment aufgebaut. Mann kan sich das wie ein Lebensmittelautomat vorstellen, der immer das selbe macht. Der Automat hat einen Schlitz, indem du einen Aktion reinwirfst. Aktion(1 Lebensmittel kaufen, 2 spuckt wieder raus weil zu wenig, 3 nichts tun). Danach kommt der Zustand(State) er beschreibt wie der Automat aussieht(Aktueller Kontostand, Preis des Lebensmittels, ob du schon was gekauft hast). Und so sieht es aus wenn ein Mensch(Agent) den Automaten(Env) benutzen würde. Agent → Aktion 1 (kaufen), Automat → "Zu wenig Geld!" → Reward -1 oder Agent → Aktion 3 (warten), Automat → "Okay, nichts passiert." → Reward 0
#### Oder
Enviroment kann man sich auch wie ein Klassenzimmer vorstellen: Regeln sind festgelegt wie test, aufgaben. Lehrer bewertet, richtig 5 punkte falsch -10. Übungen werde dort gemacht. Und am ende des Tages bekommst du ein Feedback. Kurzgefasst das Klassenzimmer speichert nicht was ich gelernt habe, es teste/bewertet mich nur. Kurzgesagt ein Ort wo gelernt wird.

### Environment Design

##### State
Ein Trading bot wie man bei der Demo version sieht braucht nicht allzu viel in State ausser Preis und Position. Warum preis und Position, wenn eine aktion passiert ändert sich der State zu State plus 1 oder minus 1. Aber wir wollen unseren Trading bot komplexer machen und den State komplexer machen, heisst wir fügen zusaätzlich zu Preis und Position Preisbezogene Features und technische Indikatoren. **Preisbezogene Features: aktueller Preis, Open, High, Low, Close, Volume**. Technische Indikatoren, das sind mathematische Werkzeuge, die Tradern Muster im Preis zeigen. **Moving Averages** (MA5, MA20, MA50…), das ist der Durchschnittspreis der letzten X Tage/Schritte. **RSI (Relative Strength Index)**, ein Wert zwischen 0 und 100, der zeigt: über 70 → überkauft (Preis vielleicht zu hoch), unter 30 → überverkauft (Preis vielleicht zu niedrig) → Zeigt, ob der Markt „gestresst“ ist. **MACD**, ein Trendfolge-Indikator mit zwei Teilen: Signal → zeigt die Trendrichtung, Histogram → zeigt Stärke / Geschwindigkeit des Trends → Gut zum Erkennen von Trendwechseln. **Bollinger Bands** (Upper, Lower), drei Linien: mittlere Linie: Durchschnitt, obere Linie: Durchschnitt + Volatilität, untere Linie: Durchschnitt – Volatilität → Sie zeigen, wie „ausgedehnt“ der Preis gerade ist. Wenn der Preis oben anstößt → vielleicht zu hoch. Wenn unten → vielleicht zu tief. **Volatility** (ATR – Average True Range), misst, wie stark sich der Preis bewegt. Hohe ATR → Markt ist wild, große Bewegungen, niedrige ATR → Markt ist ruhig → Extrem wichtig für Risiko & Positionsgrößen. Preisbezogene Daten sagen dir, wo der Preis war. Indikatoren sagen dir, was der Preis gerade fühlt (Trend, Stärke, Volatilität).

<img width="300" height="175" alt="image" src="https://github.com/user-attachments/assets/d8ce8f79-f9aa-499a-a91b-3b221312f3ad" />



##### Action Space
Diskrete Aktionen (Buy, Sell, Hold), ideal für Q-Learning
und leicht verständlich im Debugging.

##### Reward
Reward = realisierter Gewinn.
Warum nicht unrealisiert? 
→ Weil der Agent sonst risikolose Buy-and-Hold Strategien bevorzugt.

##### Episode Ending
Episode endet am Ende des Datensatzes.

### Agent
Agent gleich schüler der ins Klassenzimmer(env) geht. Mache meine Aufgaben, bekomme Punkte als Rückmeldung. Was ich lerne im Klassenzimmer(env) wird in meinem Gehirn gespeichert darauffolgend werde ich schlauer, und treffe nächstes mal bessere Entscheidungen. Kurzgesagt Agent gleich derjenige der lernt.

### Training

### Backtest‑Auswertung

### Paper‑Trading

### Optional: Vorbereitung auf Real‑Trading
---
