# Analisi oggettiva dei fronti ICON-2I (v13)

## Scopo e limite fondamentale

Il sito pubblica una **stima oggettiva e conservativa** dei fronti sinottici
nel dominio ICON-2I. La geometria proviene esclusivamente da ICON-2I; ECMWF
resta un modello di superficie separato e non conferma, modifica o sostituisce
i fronti.

La linea non è un'osservazione né un'analisi manuale del Servizio
Meteorologico. `qualityScore` e `uncertaintyIndex` descrivono la coerenza
interna delle prove in un singolo run deterministico: non sono probabilità
calibrate e non misurano l'errore previsionale assoluto.

Il metodo operativo è `icon2i-ofa-multilevel-strict-v13`.

## Dati usati

Per ogni run 00/12 UTC e per tutte le 73 scadenze orarie:

- T, QV, U e V a 850 hPa: campi obbligatori e geometria primaria;
- PMSL: firma di saccatura e tendenza barica, opzionale;
- T, QV, U e V a 925 hPa: coerenza verticale e vento più vicino al suolo,
  opzionali;
- OMEGA a 700 hPa: attività/ascesa frontale, opzionale;
- HSURF: controllo orografico.

Il livello 850 hPa riduce il rumore dello strato limite e i contrasti diurni.
Il livello 925 hPa interseca il terreno già attorno a 750 m: sopra 650 m i
campioni vengono esclusi. Dove almeno il 60% del confronto resta valido, la
coerenza termica a 925 hPa diventa una porta obbligatoria; altrove non viene
inventato un dato sostitutivo. Omega resta una prova morbida: un fronte maturo
o frontolitico può esistere senza forte ascesa istantanea.

La fonte ufficiale descrive ICON-2I come modello deterministico a circa
2,2 km, dominio 3–22°E / 33–49°N, orizzonte +72 h:
[MeteoHub — dataset ICON-2I](https://meteohub.agenziaitaliameteo.it/app/datasets).

## 1. Variabili termiche indipendenti

Da pressione, temperatura e umidità specifica si calcolano:

- temperatura potenziale secca `theta`;
- temperatura potenziale equivalente con la formulazione di Bolton;
- temperatura potenziale di bulbo umido `theta_w` con l'approssimazione di
  Davies-Jones (2008).

La geometria viene cercata separatamente sia su `theta_w` sia sulla
temperatura potenziale secca. `theta_w` distingue bene le masse d'aria umide;
la seconda ricerca recupera intrusioni fredde secche che un compenso di
umidità può rendere poco visibili in `theta_w`. I due insiemi vengono uniti e
deduplicati, ma nessuno dei due può pubblicare una linea senza superare gli
stessi controlli incrociati di temperatura secca, `theta_w` e densità
virtuale. Un confine di sola umidità viene quindi respinto.

## 2. Localizzazione OFA

Il nucleo segue Hewson (1998) nella formulazione portabile verificata da
Sansom e Catto (2024):

1. smoothing gaussiano in chilometri, prima delle derivate;
2. gradiente metrico di `theta_w`;
3. Thermal Front Locator
   `TFL = laplacian(|grad(theta_w)|)`;
4. estrazione delle sole isolinee `TFL = 0`;
5. mascheramento della linea con TFP e intensità della zona baroclina
   adiacente (ABZ).

Il Thermal Front Parameter conserva il segno standard:

```text
TFP = grad(|grad(theta_w)|) · grad(theta_w) / |grad(theta_w)|
```

Sul bordo caldo della zona baroclina è negativo. L'ABZ è la stima locale
pubblicata, non un campione arbitrario preso lontano dalla linea:

```text
ABZ = |grad(theta_w)|
      + grid_length / sqrt(2) * |grad(|grad(theta_w)|)|
```

Le derivate rispettano le distanze reali della griglia lon/lat; cambiare
risoluzione non cambia implicitamente le unità delle soglie.

## 3. Soppressione del rumore ICON-2I

Si applica un rilevatore a due scale sullo stesso modello:

- scala sinottica: sigma 100 km, derivata 30 km, linea minima 350 km;
- scala raffinata: sigma 45 km, derivata 15 km, linea minima 220 km;
- almeno il 60% della linea raffinata deve stare entro 110 km da una linea
  sinottica;
- margine di 70 km dal bordo del dominio.

La scala forte decide se esiste una struttura sinottica; la scala raffinata
ne conserva la geometria ICON-2I. Un gradiente locale molto intenso non può
aggirare il requisito sinottico: è il meccanismo principale contro brezze,
outflow convettivi, canali vallivi e cold pool.

Soglie fuzzy OFA, espresse in unità fisiche:

| Scala | TFP debole → pieno | ABZ debole → pieno |
|---|---:|---:|
| Sinottica | −0,15 → −0,40 K/(100 km)² | 0,65 → 1,10 K/100 km |
| Raffinata | −0,25 → −0,90 K/(100 km)² | 0,90 → 1,70 K/100 km |

I quantili del dominio possono rendere le soglie più severe, mai più
permissive. Vengono inoltre respinte anomalie quasi chiuse, hairpin molto
sinuosi, duplicati paralleli e linee prevalentemente sopra terreno elevato.

## 4. Controlli fisici indipendenti

Ogni linea viene orientata in modo univoco con l'aria calda a sinistra. A
circa 55 km sui due lati si misurano:

- salto di `theta_w`;
- salto di temperatura secca;
- salto di temperatura potenziale virtuale, proxy della densità;
- allineamento fra normale frontale e gradiente secco;
- rotazione del vento e convergenza normale.

Sulla linea si calcolano inoltre:

- vorticità e convergenza orizzontale;
- frontogenesi orizzontale di Petterssen/Keyser, in
  K/(100 km)/(3 h);
- avvezione termica;
- velocità frontale OFA;
- profondità della saccatura PMSL a ±100 km e tendenza isallobarica;
- supporto 925 hPa e omega 700 hPa, quando disponibili.

I requisiti duri impongono almeno un vero contrasto secco e di densità,
allineamento termico plausibile, sufficiente firma OFA e geometria sinottica.
### Porte logiche non compensabili

La v13 non decide l'esistenza di un fronte con la sola media dei punteggi.
Prima del ranking applica porte fisiche obbligatorie lungo l'intero segmento:

- contrasto simultaneo di `theta_w`, temperatura secca e densità virtuale;
- orientamento corretto del gradiente secco per almeno metà della linea;
- firma sinottica a due scale e geometria aperta/coerente;
- discontinuità vettoriale del vento (intensità oppure rotazione);
- almeno una firma dinamica fra convergenza, vorticità ciclonica e
  frontogenesi;
- rifiuto esplicito della divergenza forte non sostenuta dalla dinamica;
- rifiuto esplicito di una cresta di pressione lungo la linea;
- coerenza a 925 hPa obbligatoria dove quel livello è realmente sopra il
  terreno per una parte sufficiente del segmento.

Le statistiche sono calcolate anche come frazioni della linea: un ottimo
segnale su pochi punti non può nascondere una maggioranza incoerente.

Soltanto dopo queste porte il punteggio di evidenza ordina le linee ammesse:

- 38% termodinamica;
- 24% dinamica;
- 10% pressione;
- 10% coerenza verticale;
- 4% attività a 700 hPa;
- 14% struttura.

È un indice trasparente e diagnostico, non machine learning addestrato su
etichette e non una probabilità.

## 5. Tracciamento e tipo di fronte

I candidati orari sono associati globalmente con algoritmo ungherese usando
distanza simmetrica fra linee, orientamento, lunghezza, normale calda e
posizione prevista. Un'identità non può saltare più di due ore.

Il tipo non viene fissato una volta per l'intera vita della traccia: viene
ricalcolato in una finestra temporale locale per ogni scadenza. Lo stesso
confine può quindi passare in modo coerente da freddo a stazionario senza
sparire e senza conservare un simbolo ormai vecchio.

Per essere pubblicata una traccia deve avere:

- almeno 4 rilevamenti;
- durata di almeno 3 ore;
- copertura temporale almeno 72%;
- `qualityScore >= 0.64`;
- `uncertaintyIndex <= 0.36`;
- classificazione non conflittuale.

La classificazione confronta tre famiglie indipendenti:

1. spostamento geometrico orario verso il lato caldo/freddo;
2. velocità di fase `-d(theta_w)/dt / |grad(theta_w)|`;
3. famiglia del vento: velocità OFA
   `V · grad|grad(theta_w)| / |grad|grad(theta_w)||` e componente del vento
   normale al confine delle masse d'aria. Dove possibile si usa il vento a
   925 hPa, con fallback a 850 hPa sopra il terreno elevato.

Con soglia 1,5 m/s (5,4 km/h), moto verso l'aria calda = fronte freddo,
moto verso l'aria fredda = fronte caldo, altrimenti stazionario. Un tipo
mobile richiede almeno due famiglie concordi e nessun voto opposto. Una
contraddizione forte del vento rende la traccia `uncertain`, anche quando la
linea termica sembra muoversi correttamente.

Si applica inoltre isteresi fisica: soltanto un candidato forte può creare
una traccia; un candidato marginale ma ancora plausibile può continuarla per
non più di due ore. Non può nascere una linea da sole evidenze marginali e
una lunga perdita di segnale termina davvero l'identità.

## 6. Significato dell'incertezza

L'indice aumenta con prove fisiche deboli, scarsa continuità, moto instabile,
geometria dubbia o classificazione discordante. Il sito pubblica soltanto la
classe interna bassa. Rimangono fuori dall'indice:

- errore del run ICON-2I e sensibilità alle condizioni iniziali;
- errore sistematico del modello;
- verifica contro osservazioni, radiosondaggi e analisi manuali;
- probabilità ensemble.

Perciò “incertezza interna bassa” significa che il fronte è coerente con i
campi del run, non che la posizione reale sia garantita. Per una probabilità
meteorologica occorrerebbero ensemble e verifica retrospettiva con un archivio
di analisi ufficiali.

## 7. Test automatici

La workflow blocca la pubblicazione se falliscono i test sintetici:

- segno TFP e posizione sul bordo caldo;
- invarianza a orientamento della griglia e risoluzione;
- rimozione di patch locali e anomalie chiuse;
- traslazione rigida senza falsa frontogenesi;
- convergenza con frontogenesi positiva;
- rotazione rigida con vorticità ma senza falsa deformazione;
- rifiuto dei confini di sola umidità;
- rifiuto di gradiente termico con vento divergente/contrario;
- rifiuto di una linea collocata su una cresta barica;
- tracciamento, classificazione e separazione delle identità.
- consenso obbligatorio fra moto geometrico, fase termica e vento;
- isteresi di una sola ora senza nascita da candidati marginali.

## Riferimenti primari

- Hewson, 1998, *Objective fronts*:
  [Meteorological Applications](https://www.cambridge.org/core/journals/meteorological-applications/article/objective-fronts/DEF4E8845B0B5DBC102560C658FB4B6B)
- Sansom e Catto, 2024, *A portable objective front identification method*:
  [Geoscientific Model Development](https://gmd.copernicus.org/articles/17/6137/2024/gmd-17-6137-2024.html)
- Codice ufficiale collegato all'articolo:
  [phil-sansom/front_id](https://github.com/phil-sansom/front_id)
- Beckert et al., 2023, rilevamento tridimensionale e filtri fuzzy:
  [Geoscientific Model Development](https://gmd.copernicus.org/articles/16/4427/2023/gmd-16-4427-2023.html)
- Jenkner et al., 2010, fronti ad alta risoluzione e orografia alpina:
  [Meteorological Applications](https://rmets.onlinelibrary.wiley.com/doi/abs/10.1002/met.142)
- Bitsa et al., 2021, criteri termo-dinamici nel Mediterraneo:
  [International Journal of Climatology](https://rmets.onlinelibrary.wiley.com/doi/10.1002/joc.7208)
- Davies-Jones, 2008, temperatura potenziale di bulbo umido:
  [Monthly Weather Review](https://journals.ametsoc.org/view/journals/mwre/136/7/2007mwr2224_1.xml)
