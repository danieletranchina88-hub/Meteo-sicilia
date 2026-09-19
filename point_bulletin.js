(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.MeteoPointBulletin = api;
}(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const REQUIRED_FIELDS = [
    "temperature2m", "feelsLike", "rainStep", "pressureMsl",
    "relativeHumidity2m", "cloudCover", "windU10", "windV10",
    "convectionProbability", "capeMl", "cinMl", "visibility",
    "stormConfidence", "fogProbability", "freezingRainRisk", "foehnIndex",
    "frontDistanceKm"
  ];

  function finite(value) {
    return value !== null && value !== undefined && value !== "" &&
      Number.isFinite(Number(value));
  }

  function values(series, name) {
    return Array.isArray(series && series[name])
      ? series[name].map(value => finite(value) ? Number(value) : null)
      : [];
  }

  function validValues(input) {
    return (input || []).filter(finite).map(Number);
  }

  function sum(input) {
    return validValues(input).reduce((total, value) => total + value, 0);
  }

  function average(input) {
    const valid = validValues(input);
    return valid.length
      ? valid.reduce((total, value) => total + value, 0) / valid.length
      : null;
  }

  function percentile(input, percentileValue) {
    const valid = validValues(input).sort((a, b) => a - b);
    if (!valid.length) return null;
    const position = (valid.length - 1) * percentileValue;
    const lower = Math.floor(position);
    const fraction = position - lower;
    return valid[lower + 1] === undefined
      ? valid[lower]
      : valid[lower] + fraction * (valid[lower + 1] - valid[lower]);
  }

  function extreme(input, mode) {
    let selected = null;
    (input || []).forEach((value, index) => {
      if (!finite(value)) return;
      if (!selected || (mode === "min"
        ? Number(value) < selected.value
        : Number(value) > selected.value)) {
        selected = { value: Number(value), index };
      }
    });
    return selected;
  }

  function timeAt(series, index, options) {
    const payload = series && series.times && series.times[index];
    if (!payload || !payload.validTime) return "orario non disponibile";
    return new Intl.DateTimeFormat("it-IT", {
      weekday: options && options.weekday ? "long" : undefined,
      day: "2-digit",
      month: options && options.month ? "short" : undefined,
      hour: "2-digit",
      minute: "2-digit"
    }).format(new Date(payload.validTime));
  }

  function rangeLabel(series) {
    if (!series || !series.times || !series.times.length) {
      return "orizzonte non disponibile";
    }
    return timeAt(series, 0, { month: true }) + " – " +
      timeAt(series, series.times.length - 1, { month: true });
  }

  function windSpeed(u, v) {
    const count = Math.max(u.length, v.length);
    return Array.from({ length: count }, (_, index) =>
      finite(u[index]) && finite(v[index])
        ? Math.hypot(Number(u[index]), Number(v[index])) * 3.6
        : null
    );
  }

  function windFromDegrees(u, v) {
    if (!finite(u) || !finite(v)) return null;
    return (Math.atan2(Number(u), Number(v)) * 180 / Math.PI + 180 + 360) % 360;
  }

  function compass(degrees) {
    if (!finite(degrees)) return "";
    return ["N", "NE", "E", "SE", "S", "SO", "O", "NO"][
      Math.round(Number(degrees) / 45) % 8
    ];
  }

  function firstIndex(input, predicate) {
    for (let index = 0; index < (input || []).length; index += 1) {
      if (finite(input[index]) && predicate(Number(input[index]))) return index;
    }
    return -1;
  }

  function lastIndex(input, predicate) {
    for (let index = (input || []).length - 1; index >= 0; index -= 1) {
      if (finite(input[index]) && predicate(Number(input[index]))) return index;
    }
    return -1;
  }

  function consecutiveMaximum(input, predicate) {
    let best = 0;
    let current = 0;
    (input || []).forEach(value => {
      current = finite(value) && predicate(Number(value)) ? current + 1 : 0;
      best = Math.max(best, current);
    });
    return best;
  }

  function episodes(input, predicate) {
    const result = [];
    let current = null;
    (input || []).forEach((value, index) => {
      const active = finite(value) && predicate(Number(value));
      if (active) {
        if (!current) current = { start: index, end: index, total: 0, peak: null };
        current.end = index;
        current.total += Number(value);
        if (!current.peak || Number(value) > current.peak.value) {
          current.peak = { value: Number(value), index: index };
        }
      } else if (current) {
        result.push(current);
        current = null;
      }
    });
    if (current) result.push(current);
    return result;
  }

  function windowExtreme(input, center, radius, mode) {
    if (!Number.isInteger(center) || center < 0) return null;
    let selected = null;
    const start = Math.max(0, center - radius);
    const end = Math.min((input || []).length - 1, center + radius);
    for (let index = start; index <= end; index += 1) {
      const value = input[index];
      if (!finite(value)) continue;
      if (!selected || (mode === "min"
        ? Number(value) < selected.value
        : Number(value) > selected.value)) {
        selected = { value: Number(value), index };
      }
    }
    return selected;
  }

  function number(value, decimals) {
    return finite(value) ? Number(value).toFixed(decimals) : "—";
  }

  function dailyEvolution(series, dataSeries) {
    const groups = [];
    (series && series.times || []).forEach((time, index) => {
      if (!time || !time.validTime) return;
      const date = new Date(time.validTime);
      if (Number.isNaN(date.getTime())) return;
      const key = date.toLocaleDateString("sv-SE");
      let group = groups.find(item => item.key === key);
      if (!group) {
        group = { key, date, indexes: [] };
        groups.push(group);
      }
      group.indexes.push(index);
    });
    return groups.slice(0, 4).map(group => {
      const select = name => group.indexes.map(index => dataSeries[name][index]);
      const minimum = extreme(select("temperature"), "min");
      const maximum = extreme(select("temperature"), "max");
      const rainfall = sum(select("rain"));
      const convection = extreme(select("convection"), "max");
      const wind = extreme(select("wind"), "max");
      const cloudiness = average(select("cloud"));
      const label = new Intl.DateTimeFormat("it-IT", {
        weekday: "long", day: "2-digit", month: "short"
      }).format(group.date);
      let weather;
      if (convection && convection.value >= 70) weather = "forte segnale diagnostico temporalesco";
      else if (convection && convection.value >= 40) weather = "segnale diagnostico per rovesci o temporali locali";
      else if (rainfall >= 10) weather = "piogge a tratti diffuse";
      else if (rainfall >= 0.5) weather = "qualche pioggia o rovescio";
      else if (cloudiness !== null && cloudiness >= 75) weather = "molte nubi, ma fenomeni scarsi";
      else if (cloudiness !== null && cloudiness <= 30) weather = "tempo in prevalenza soleggiato e asciutto";
      else weather = "nuvolosità variabile e tempo perlopiù asciutto";

      let text = `${label}: ${weather}`;
      if (minimum && maximum) {
        text += `, temperature ${number(minimum.value, 0)}/${number(maximum.value, 0)} °C`;
      }
      if (rainfall >= 0.1) text += `, accumulo circa ${number(rainfall, 1)} mm`;
      if (wind && wind.value >= 30) text += `, vento fino a ${number(wind.value, 0)} km/h`;
      return text + ".";
    });
  }

  function buildHeadline(data) {
    if (data.convectionMax && data.convectionMax.value >= 70) {
      if (data.stormConfidenceAtPeak !== null && data.stormConfidenceAtPeak < 45) {
        return "Il modello propone una finestra temporalesca importante, ma con bassa coerenza interna: è un segnale da verificare nei prossimi aggiornamenti.";
      }
      return "Fase potenzialmente temporalesca, con una finestra di innesco marcata nel corso delle prossime 72 ore.";
    }
    if (data.rainTotal >= 20) {
      return "Previsione caratterizzata da una fase piovosa significativa, intervallata da pause più asciutte.";
    }
    if (data.windMax && data.windMax.value >= 55) {
      return "Il vento rappresenta l’elemento più rilevante della previsione, con rinforzi localmente sostenuti.";
    }
    if (data.fogMax && data.fogMax.value >= 60) {
      return "Tempo nel complesso poco perturbato, ma con una finestra favorevole a nebbia o forte riduzione della visibilità.";
    }
    if (data.rainTotal >= 1) {
      return "Evoluzione variabile, con passaggi piovosi locali e condizioni non uniformi nell’arco delle 72 ore.";
    }
    if (data.cloudMean !== null && data.cloudMean <= 35) {
      return "Previsione prevalentemente stabile e asciutta, con nuvolosità generalmente contenuta.";
    }
    return "Previsione in prevalenza asciutta, con nuvolosità variabile e senza segnali di maltempo organizzato.";
  }

  function classifyCAPERegime(capeMax, cinNear) {
    if (!capeMax || !finite(capeMax.value)) return null;
    var cape = capeMax.value;
    var cin = cinNear && finite(cinNear.value) ? cinNear.value : 0;
    if (cape >= 2500) return { level: "estrema", desc: "energia convettiva disponibile molto elevata (ML-CAPE > 2500 J/kg), capace di sostenere convezione intensa se innesco e organizzazione sono presenti" };
    if (cape >= 1500) return { level: "forte", desc: "energia convettiva significativa (ML-CAPE " + number(cape, 0) + " J/kg), sufficiente a sostenere celle intense quando gli altri ingredienti coincidono" };
    if (cape >= 800) return { level: "moderata", desc: "instabilità latente moderata (ML-CAPE " + number(cape, 0) + " J/kg), che in presenza di forzanti dinamiche può produrre temporali localmente intensi" };
    if (cape >= 300) return { level: "debole", desc: "modesta energia convettiva (ML-CAPE " + number(cape, 0) + " J/kg), con innesco probabile solo in presenza di convergenza marcata al suolo" };
    return { level: "trascurabile", desc: "energia convettiva trascurabile (ML-CAPE < 300 J/kg), situazione non favorevole a temporali" };
  }

  function buildThermodynamicAnalysis(data) {
    var lines = [];
    var capeRegime = classifyCAPERegime(data.capeMax, data.cinNearConvection);
    if (!capeRegime) return null;

    lines.push("Analisi termodinamica: " + capeRegime.desc + ".");

    if (data.cinNearConvection && finite(data.cinNearConvection.value)) {
      var cin = data.cinNearConvection.value;
      if (cin < -200) {
        lines.push("L'inibizione convettiva è molto forte (CIN " + number(cin, 0) + " J/kg): il coperchio è solido e l'innesco richiederebbe forzanti eccezionali, nonostante l'energia disponibile.");
      } else if (cin < -100) {
        lines.push("L'inibizione convettiva è significativa (CIN " + number(cin, 0) + " J/kg): l'innesco è sfavorito senza un sollevamento meccanico robusto, ad esempio forzato da fronti o convergenza orografica.");
      } else if (cin < -50) {
        lines.push("L'inibizione è moderata (CIN " + number(cin, 0) + " J/kg): sufficiente a ritardare l'innesco ma non a impedirlo in presenza di riscaldamento diurno e convergenze locali.");
      } else {
        lines.push("L'inibizione è debole o assente (CIN " + number(cin, 0) + " J/kg): l'atmosfera è prossima alla soglia di innesco libero.");
      }
    }

    if (data.convectionMax && finite(data.convectionMax.value) && data.convectionMax.value >= 40) {
      var prob = data.convectionMax.value;
      if (prob >= 70) {
        lines.push("La diagnostica integrata, che verifica anche LPI, updraft, umidità, innesco e persistenza temporale, raggiunge " + number(prob, 0) + "/100.");
      } else {
        lines.push("L'algoritmo composito raggiunge " + number(prob, 0) + "/100: è uno score deterministico di supporto temporalesco, non una frequenza osservata.");
      }
    }

    return lines.join(" ");
  }

  function buildPressureAnalysis(data, pressure) {
    if (!data.pressureMin || !finite(data.pressureMin.value)) return null;
    var lines = [];
    var pressMin = data.pressureMin.value;

    // Pressure tendency over first 12h
    var trend12h = null;
    if (pressure.length >= 13 && finite(pressure[0]) && finite(pressure[12])) {
      trend12h = pressure[12] - pressure[0];
    }

    if (trend12h !== null && Math.abs(trend12h) >= 2) {
      if (trend12h < -4) {
        lines.push("Campo barico: la MSLP cala rapidamente (" + number(Math.abs(trend12h), 1) + " hPa in 12 ore). Il segnale è compatibile con l'avvicinamento o l'approfondimento di una struttura depressionaria, da verificare con vento, fronti e precipitazione.");
      } else if (trend12h < -2) {
        lines.push("Campo barico: la MSLP diminuisce di " + number(Math.abs(trend12h), 1) + " hPa in 12 ore; la sola tendenza barica non identifica il tipo né l'intensità del sistema in avvicinamento.");
      } else if (trend12h > 4) {
        lines.push("Campo barico: la MSLP risale nettamente (" + number(trend12h, 1) + " hPa in 12 ore). È un segnale di aumento della pressione, ma non dimostra da solo un passaggio frontale o l'arrivo di schiarite.");
      } else {
        lines.push("Campo barico: la MSLP aumenta gradualmente di " + number(trend12h, 1) + " hPa in 12 ore; l'evoluzione del tempo va letta insieme agli altri campi.");
      }
    } else if (pressMin < 1000) {
      lines.push("Campo barico: minimo puntuale di " + number(pressMin, 0) + " hPa. Il valore colloca il punto in un contesto di pressione relativamente bassa, ma non identifica da solo un ciclone o un fronte attivo.");
    }

    return lines.length ? lines.join(" ") : null;
  }

  function buildFrontAnalysis(data, series) {
    if (!data.frontMin || !finite(data.frontMin.value) || data.frontMin.value > 150) return null;
    var dist = data.frontMin.value;
    var lines = [];

    if (dist <= 30) {
      lines.push("La linea frontale oggettiva passa entro circa " + number(dist, 0) + " km dal punto. Rotazione del vento, variazione termica e precipitazioni sono segnali da verificare nei campi dedicati, non conseguenze automatiche della distanza.");
    } else if (dist <= 80) {
      lines.push("La linea frontale oggettiva più vicina si porta a circa " + number(dist, 0) + " km. Il punto può rientrare nella fascia frontale o nella sua incertezza posizionale, ma il coinvolgimento va confermato con vento, θw e precipitazione.");
    } else {
      lines.push("La linea frontale più vicina resta a circa " + number(dist, 0) + " km: è informazione di contesto sinottico e la sola distanza non consente di attribuire fenomeni al punto.");
    }

    return lines.join(" ");
  }

  function buildPointBulletin(series, locationName) {
    const location = locationName || "zona selezionata";
    const temperature = values(series, "temperature2m");
    const feelsLike = values(series, "feelsLike");
    const rain = values(series, "rainStep");
    const pressure = values(series, "pressureMsl");
    const humidity = values(series, "relativeHumidity2m");
    const cloud = values(series, "cloudCover");
    const uWind = values(series, "windU10");
    const vWind = values(series, "windV10");
    const wind = windSpeed(uWind, vWind);
    const convection = values(series, "convectionProbability");
    const cape = values(series, "capeMl");
    const cin = values(series, "cinMl");
    const visibility = values(series, "visibility");
    const fog = values(series, "fogProbability");
    const freezing = values(series, "freezingRainRisk");
    const foehn = values(series, "foehnIndex");
    const frontDistance = values(series, "frontDistanceKm");
    const stormConfidence = values(series, "stormConfidence");
    const stormContradiction = values(series, "stormContradiction");
    const gust = values(series, "windGust10").map(value =>
      finite(value) ? Number(value) * 3.6 : null
    );

    const data = {
      temperatureMin: extreme(temperature, "min"),
      temperatureMax: extreme(temperature, "max"),
      feelsMax: extreme(feelsLike, "max"),
      rainTotal: sum(rain),
      rainMax: extreme(rain, "max"),
      rainStart: firstIndex(rain, value => value >= 0.1),
      rainEnd: lastIndex(rain, value => value >= 0.1),
      rainEpisodes: episodes(rain, value => value >= 0.1),
      wetHours: validValues(rain).filter(value => value >= 0.1).length,
      convectionMax: extreme(convection, "max"),
      stormEpisodes: episodes(convection, value => value >= 40),
      stormHighDuration: consecutiveMaximum(convection, value => value >= 70),
      capeMax: extreme(cape, "max"),
      windMax: extreme(wind, "max"),
      gustMax: extreme(gust, "max"),
      pressureMin: extreme(pressure, "min"),
      fogMax: extreme(fog, "max"),
      visibilityMin: extreme(visibility, "min"),
      frontMin: extreme(frontDistance, "min"),
      cloudMean: average(cloud),
      humidityMean: average(humidity),
      freezingMax: extreme(freezing, "max"),
      foehnMax: extreme(foehn, "max")
    };
    data.capeNearConvection = data.convectionMax
      ? windowExtreme(cape, data.convectionMax.index, 3, "max") : null;
    data.cinNearConvection = data.convectionMax
      ? windowExtreme(cin, data.convectionMax.index, 3, "max") : null;
    data.stormConfidenceAtPeak = data.convectionMax && finite(
      stormConfidence[data.convectionMax.index]
    ) ? Number(stormConfidence[data.convectionMax.index]) : null;
    data.stormContradictionAtPeak = data.convectionMax && finite(
      stormContradiction[data.convectionMax.index]
    ) ? Number(stormContradiction[data.convectionMax.index]) : null;

    const paragraphs = [];
    const sections = [];
    function addSection(title, text) {
      if (!text) return;
      sections.push({ title: title, text: text });
      paragraphs.push(title + " — " + text);
    }
    const headline = buildHeadline(data);
    addSection("Sintesi", headline);
    const evolution = dailyEvolution(series, {
      temperature, rain, convection, wind, cloud
    });
    if (evolution.length) {
      addSection("Evoluzione giorno per giorno", evolution.join(" "));
    }

    if (data.rainTotal >= 0.1 && data.rainStart >= 0) {
      let rainText = `Sono stimati complessivamente ${number(data.rainTotal, 1)} mm`;
      rainText += ` distribuiti su circa ${data.wetHours} ore con precipitazione.`;
      if (data.rainMax && data.rainMax.value >= 0.5) {
        rainText += ` Il passaggio più intenso è previsto attorno a ${timeAt(series, data.rainMax.index)}, con ${number(data.rainMax.value, 1)} mm nel passo orario.`;
      }
      rainText += ` La prima fase utile compare da ${timeAt(series, data.rainStart)}`;
      if (data.rainEnd > data.rainStart) {
        rainText += `, con gli ultimi segnali entro ${timeAt(series, data.rainEnd)}`;
      }
      rainText += ".";
      const mainEpisode = data.rainEpisodes.reduce((best, episode) =>
        !best || episode.total > best.total ? episode : best, null
      );
      if (mainEpisode && data.rainEpisodes.length > 1) {
        rainText += ` I periodi piovosi distinti sono ${data.rainEpisodes.length}; il principale va da ${timeAt(series, mainEpisode.start)} a ${timeAt(series, mainEpisode.end)}.`;
      }
      addSection("Precipitazioni", rainText);
    } else {
      addSection("Precipitazioni", "Non emergono accumuli di pioggia significativi nel punto considerato; eventuali fenomeni molto locali possono comunque sfuggire alla griglia del modello.");
    }

    if (data.convectionMax) {
      if (data.convectionMax.value >= 70) {
        let convectionText =
          `L'indice diagnostico temporalesco raggiunge ${number(data.convectionMax.value, 0)}/100 attorno a ${timeAt(series, data.convectionMax.index)}.`;
        if (data.capeNearConvection) {
          convectionText +=
            ` Nella finestra di ±3 ore il ML-CAPE arriva fino a ${number(data.capeNearConvection.value, 0)} J/kg`;
          if (data.cinNearConvection) {
            convectionText += ` e il ML-CIN si attenua fino a ${number(data.cinNearConvection.value, 0)} J/kg`;
          }
          convectionText += ".";
        }
        if (data.stormConfidenceAtPeak !== null) {
          convectionText += ` La coerenza e copertura interne in quell’ora sono ${number(data.stormConfidenceAtPeak, 0)}/100`;
          if (data.stormContradictionAtPeak !== null && data.stormContradictionAtPeak >= 30) {
            convectionText += `, con contraddizioni interne ${number(data.stormContradictionAtPeak, 0)}/100`;
          }
          convectionText += ".";
        }
        if (data.stormHighDuration >= 2) {
          convectionText += ` La soglia 70/100 persiste per ${data.stormHighDuration} ore consecutive.`;
        }
        convectionText +=
          " Lo score non è calibrato su fulmini osservati e non garantisce che una cella colpisca il punto.";
        addSection("Temporali", convectionText);
      } else if (data.convectionMax.value >= 40) {
        addSection("Temporali",
          `È presente una finestra di supporto convettivo moderato attorno a ${timeAt(series, data.convectionMax.index)}, con indice temporalesco massimo ${number(data.convectionMax.value, 0)}/100. ` +
          "Il valore descrive una diagnostica deterministica, non la frequenza attesa dei temporali."
        );
      } else {
        addSection("Temporali",
          `Il segnale temporalesco rimane basso: il massimo puntuale è ${number(data.convectionMax.value, 0)}/100, previsto attorno a ${timeAt(series, data.convectionMax.index)}.`
        );
      }
    }

    const thermoAnalysis = buildThermodynamicAnalysis(data);
    if (thermoAnalysis) addSection("Profilo convettivo", thermoAnalysis.replace(/^Analisi termodinamica:\s*/, ""));

    const pressureAnalysis = buildPressureAnalysis(data, pressure);
    if (pressureAnalysis) addSection("Pressione", pressureAnalysis.replace(/^Campo barico:\s*/, ""));

    const frontAnalysis = buildFrontAnalysis(data, series);
    if (frontAnalysis) addSection("Fronti", frontAnalysis);

    if (data.temperatureMin && data.temperatureMax) {
      let temperatureText =
        `Le temperature oscilleranno tra ${number(data.temperatureMin.value, 1)} °C, attesi ${timeAt(series, data.temperatureMin.index)}, ` +
        `e ${number(data.temperatureMax.value, 1)} °C, previsti ${timeAt(series, data.temperatureMax.index)}.`;
      if (data.feelsMax && data.feelsMax.value - data.temperatureMax.value >= 2) {
        temperatureText += ` Nelle ore più calde la temperatura percepita potrà raggiungere ${number(data.feelsMax.value, 1)} °C.`;
      }
      if (data.humidityMean !== null) {
        temperatureText += ` L’umidità media si manterrà intorno al ${number(data.humidityMean, 0)}%.`;
      }
      addSection("Temperature", temperatureText);
    }

    if (data.windMax) {
      const direction = compass(windFromDegrees(
        uWind[data.windMax.index], vWind[data.windMax.index]
      ));
      let windText =
        `Venti generalmente ${data.windMax.value >= 55 ? "sostenuti" : data.windMax.value >= 30 ? "moderati, a tratti tesi" : "deboli o moderati"}, ` +
        `con un massimo di circa ${number(data.windMax.value, 0)} km/h`;
      if (direction) windText += ` da ${direction}`;
      windText += ` attorno a ${timeAt(series, data.windMax.index)}.`;
      if (pressure.length >= 7 && finite(pressure[0]) && finite(pressure[6])) {
        const trend = pressure[6] - pressure[0];
        if (Math.abs(trend) >= 1.5) {
          windText += ` Nelle prime sei ore la pressione è prevista in ${trend < 0 ? "diminuzione" : "aumento"} di circa ${number(Math.abs(trend), 1)} hPa.`;
        }
      }
      if (data.pressureMin) {
        windText += ` Il minimo barico puntuale è di ${number(data.pressureMin.value, 0)} hPa verso ${timeAt(series, data.pressureMin.index)}.`;
      }
      if (data.gustMax && data.gustMax.value > data.windMax.value + 8) {
        windText += ` Le raffiche possono raggiungere circa ${number(data.gustMax.value, 0)} km/h verso ${timeAt(series, data.gustMax.index)}.`;
      }
      addSection("Vento", windText);
    }

    const visibilityRisk = data.visibilityMin && data.visibilityMin.value < 5000;
    const fogRisk = data.fogMax && data.fogMax.value >= 35;
    if (visibilityRisk || fogRisk) {
      let text = "Attenzione alla visibilità";
      if (data.visibilityMin) {
        text += `, che potrebbe scendere fino a circa ${number(data.visibilityMin.value / 1000, 1)} km attorno a ${timeAt(series, data.visibilityMin.index)}`;
      }
      if (data.fogMax && data.fogMax.value >= 35) {
        text += `. L'indice diagnostico di nebbia raggiunge ${number(data.fogMax.value, 0)}/100`;
      }
      addSection("Visibilità", text + ".");
    }

    const hazards = [];
    if (data.freezingMax && data.freezingMax.value >= 1) {
      hazards.push("profilo favorevole a precipitazione congelantesi");
    }
    if (data.foehnMax && data.foehnMax.value >= 1) {
      hazards.push("segnale di foehn");
    }
    if (hazards.length) {
      addSection("Altri rischi", "Compare " + hazards.join(" e ") + "; il segnale va verificato con gli aggiornamenti successivi.");
    }

    const available = REQUIRED_FIELDS.filter(name =>
      validValues(values(series, name)).length > 0
    );
    const completeness = available.length / REQUIRED_FIELDS.length;
    return {
      schemaVersion: 1,
      method: "icon2i-point-timeseries-nlg-v3-scientific",
      title: "Bollettino per " + location,
      validity: rangeLabel(series),
      headline: headline,
      paragraphs,
      sections,
      dataCoverage: completeness >= 0.9 ? "alta" : completeness >= 0.65 ? "media" : "limitata",
      confidence: completeness >= 0.9 ? "alta" : completeness >= 0.65 ? "media" : "limitata",
      confidenceSemantics: "data-completeness-only-not-forecast-skill",
      completeness: Math.round(completeness * 100),
      availableFields: available,
      metrics: {
        rainTotalMm: Number(number(data.rainTotal, 1)),
        temperatureMinimumC: data.temperatureMin && data.temperatureMin.value,
        temperatureMaximumC: data.temperatureMax && data.temperatureMax.value,
        windMaximumKmh: data.windMax && data.windMax.value,
        convectionMaximumScore: data.convectionMax && data.convectionMax.value
      },
      disclaimer: "Sintesi automatica di un singolo run ICON-2I. Completezza dei campi e coerenza interna non misurano lo skill previsionale; il prodotto non è un’allerta né un bollettino ufficiale."
    };
  }

  return { buildPointBulletin };
}));
