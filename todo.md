# 📋 Diplomovka

## 🎯 ToDo
- [X] **Two HandAnnotation instances** Spraviť separátne anotovanie pre webcam a pre reference video.
- [X] **Save landmarks** Ukladanie detegovaných bodov z rúk z reference videa.
- [X] **Sampling gestures** Vyriešiť ako často sa budú ukladať ladmarky.
- [X] **Save landmarks** Ukladanie detegovaných bodov z rúk z webky.
- [X] **Load landmarks** Načítať body ak už existujú a uložiť ich.
- [ ] **Scoring** Spraviť vzdialenostné funkcie.

---

## 📅 Big picture ToDo

### Phase 1: Oprava technických chýb (Video & UI)
- [x] **Opraviť VideoWriter pre Linux** Prejsť na kodek MJPG a formát .avi kvôli kompatibilite.
- [x] **Zabezpečiť cesty** Implementovať os.path.abspath a os.makedirs pre spoľahlivé ukladanie anotovaných videí.
- [x] **Optimalizovať display** Upraviť displayReferenceVideo, aby načítavalo vopred pripravené video bez opätovného spúšťania MediaPipe v každom snímku.
- [?] **Korektné ukončenie** Pridať out.release() do eventu zatvorenia okna, aby sa nahrávka z webkamery nepoškodila.

### Phase 2: Pred-spracovanie dát (Preprocessing)
- [X] **Landmark Extraction** Vytvoriť funkciu na extrakciu všetkých 21 bodov do numpy poľa.
- [X] **Normalizácia (Translácia)** Odpočítať súradnice zápästia (ID 0) od všetkých ostatných bodov.
- [X] **Normalizácia (Mierka)** Implementovať škálovanie podľa veľkosti dlane, aby sa vyrovnali rozdiely vo vzdialenosti od kamery.
- [?] **Vektorizácia** Previesť body na vektory reprezentujúce kosti prstov.

### Phase 3: Porovnávacie algoritmy (Similarity)
- [ ] **Statické porovnanie** Implementovať Kosínovú podobnosť pre porovnanie dvoch konkrétnych póz.
- [ ] **Euklidovská vzdialenosť** Pridať výpočet $L_2$ normy pre kontrolu presnej polohy prstov.
- [ ] **Dynamické porovnanie (Sliding Window)** Vytvoriť buffer (napr. collections.deque) na ukladanie posledných 30 snímkov z webkamery.
- [ ] **Časová synchronizácia** Implementovať DTW (Dynamic Time Warping) na porovnanie pohybu používateľa s referenciou pri rôznych rýchlostiach.

### Phase 4: Spätná väzba & UI
- [ ] **Live Scoring** Prepojiť vypočítanú podobnosť s textom Score: xx% v GUI.
- [ ] **Vizuálna odozva** Zmeniť farbu textu skóre (zelená pri zhode, červená pri chybe).
- [ ] **Logovanie** Implementovať farebné výpisy do konzoly pre ľahší debugging (ANSI farby).

