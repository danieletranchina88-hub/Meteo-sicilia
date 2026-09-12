# Interfaccia e navigazione temporale

Il rinnovamento conserva i calcoli e i prodotti meteorologici esistenti e separa la previsione dalle osservazioni satellitari.

## Interfaccia

- Campi principali nella barra mobile o nei comandi laterali desktop, con un solo pulsante DOM per campo.
- Menu avanzato chiuso all'avvio, ricerca nel catalogo locale dei luoghi e chiusura con Escape.
- Vista satellite esclusiva: spegne le analisi previste e conserva i pixel originali IR/VIS/RGB. Tornando alla previsione ripristina campo, analisi e ora scelti.
- Meteogrammi: previsione al suolo in primo piano, bollettino richiudibile, diagnostiche negli altri tab, cursore orario condiviso e navigazione da tastiera.

## Cambio di ora

1. Al massimo tre download di fondo, uno con Save-Data attivo. Priorità alle ore vicine e richieste condivise fra navigazione e precaricamento.
2. Decodifica binaria e campi derivati in un Worker, con trasferimento dei buffer al client.
3. Preparazione del campo visibile in un Worker indipendente. Cache dei pixel di 64 MiB su mobile e 128 MiB su desktop; risoluzione di visualizzazione adattata per contenere la sequenza attiva. Le griglie scientifiche originali rimangono inalterate.
4. Pubblicazione tramite CanvasSource, eliminando codifica PNG, decodifica immagine e URL temporanei a ogni ora.
5. Un lavoro raster attivo e uno successivo, sempre il più recente. Le risposte superate non sostituiscono l'ora richiesta; l'orario visibile cambia insieme alla pubblicazione del campo.

Con dati e fotogramma in cache, il cambio non scarica dati né ricalcola i pixel. Il primo download, una nuova zona, un altro campo, una cache svuotata e le analisi con correzione del rilievo possono richiedere altro lavoro. Isobare e vettori conservano calcoli e cache dedicati: non è garantita latenza zero per qualsiasi combinazione di analisi, dispositivo e connessione. Il cache delle griglie del run preesistente resta in memoria; il budget sopra riguarda i fotogrammi.

## Verifiche

- `node scripts/tests/test_forecast_ui.js`: richieste condivise, ripresa dopo errori, cambi di run, risposte fuori ordine, cambio campo durante il download, coda del Worker, pubblicazione canvas, dipendenze della decodifica, riuso dei pixel, separazione satellite e unicità dei controlli.
- `node scripts/tests/test_map_3d.js`: regressioni cartografiche, metodi, dominio, particelle e correzione del rilievo; aggiornate le verifiche dei comportamenti sostituiti.
- `node scripts/tests/test_pressure_centers.js`: casi sintetici di depressioni, anticicloni, rumore e lisciamento. Il caso su un file PMSL reale è saltato quando il file non è presente.
- Verifica visiva dei controlli e dei meteogrammi con dati pubblicati. Il browser remoto della sessione non dispone di WebGL: nessuna misura FPS o verifica GPU completa della mappa.

Nessun servizio a pagamento o nuova dipendenza di produzione. Il workflow copia i quattro nuovi asset JS/CSS e avvia anche i test della navigazione temporale.
