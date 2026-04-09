# 📋 Diplomovka

## 🎯 ToDo
- [X] **Two HandAnnotation instances** Spraviť separátne anotovanie pre webcam 
a pre reference video.
- [X] **Save landmarks** Ukladanie detegovaných bodov z rúk z reference videa.
- [X] **Sampling gestures** Vyriešiť ako často sa budú ukladať ladmarky.
- [X] **Save landmarks** Ukladanie detegovaných bodov z rúk z webky.
- [X] **Load landmarks** Načítať body ak už existujú a uložiť ich.
- [X] **Scoring** Spraviť vzdialenostné funkcie.
- [ ] **Improve accuracy** Zlepšiť hodnotenie.
- [ ] **Text feedback** Navádzacie vety pre usera ako zlepšiť posunok.
- [ ] **UI** Zlepšiť UI, pridať tlačitka.
- [ ] **Video experience** Orezať videá na rozumne dlhé. Odrezať hluché časti.
- [ ] **Zohnať videá**
- [ ] **Next video** Po niekoľko percentách sa prehodí ďalšie video.


- [ ] **Control gestures** Gestá spojené s buttonmi na ovládacom paneli.

---

## 📅 Big picture ToDo

### Phase 1: Oprava technických chýb (Video & UI)
- [x] **Opraviť VideoWriter pre Linux** Prejsť na kodek MJPG a formát .avi kvôli kompatibilite.
- [x] **Zabezpečiť cesty** Implementovať os.path.abspath a os.makedirs pre spoľahlivé ukladanie 
anotovaných videí.
- [x] **Optimalizovať display** Upraviť displayReferenceVideo, aby načítavalo vopred pripravené 
video bez opätovného spúšťania MediaPipe v každom snímku.
- [?] **Korektné ukončenie** Pridať out.release() do eventu zatvorenia okna, aby sa nahrávka z 
webkamery nepoškodila.

### Phase 2: Pred-spracovanie dát (Preprocessing)
- [X] **Landmark Extraction** Vytvoriť funkciu na extrakciu všetkých 21 bodov do numpy poľa.
- [X] **Normalizácia (Translácia)** Odpočítať súradnice zápästia (ID 0) od všetkých ostatných 
bodov.
- [X] **Normalizácia (Mierka)** Implementovať škálovanie podľa veľkosti dlane, aby sa vyrovnali 
rozdiely vo vzdialenosti od kamery.
- [?] **Vektorizácia** Previesť body na vektory reprezentujúce kosti prstov.

### Phase 3: Porovnávacie algoritmy (Similarity)
- [?] **Statické porovnanie** Implementovať Kosínovú podobnosť pre porovnanie dvoch konkrétnych 
póz.
- [X] **Euklidovská vzdialenosť** Pridať výpočet $L_2$ normy pre kontrolu presnej polohy prstov.
- [X] **Dynamické porovnanie (Sliding Window)** Vytvoriť buffer (napr. collections.deque) na 
ukladanie posledných 30 snímkov z webkamery.
- [X] **Časová synchronizácia** Implementovať DTW (Dynamic Time Warping) na porovnanie pohybu 
používateľa s referenciou pri rôznych rýchlostiach.

### Phase 4: Spätná väzba & UI
- [X] **Live Scoring** Prepojiť vypočítanú podobnosť s textom Score: xx% v GUI.
- [ ] **Vizuálna odozva** Zmeniť farbu textu skóre (zelená pri zhode, červená pri chybe).
- [X] **Logovanie** Implementovať farebné výpisy do konzoly pre ľahší debugging (ANSI farby).

## Notes

Zatiaľ som sa nedostala na lepšie skóre ako 51%. Ale ak držím celý čas ruky dole tak je to malé 
skóre a keď nimi hýbem tak je to medzi 30-50%.
