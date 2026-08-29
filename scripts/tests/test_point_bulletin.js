"use strict";

const assert = require("node:assert/strict");
const { buildPointBulletin } = require("../../point_bulletin.js");

const hours = Array.from({ length: 73 }, (_, index) => ({
  leadHours: index,
  validTime: new Date(Date.UTC(2026, 6, 27, index)).toISOString()
}));
const wave = (base, amplitude) => hours.map((_, index) =>
  base + amplitude * Math.sin(index * Math.PI / 12)
);
const zeros = () => hours.map(() => 0);

const stormSeries = {
  times: hours,
  temperature2m: wave(24, 6),
  feelsLike: wave(25, 7),
  rainStep: hours.map((_, index) => index >= 17 && index <= 22 ? 2.5 : 0),
  pressureMsl: hours.map((_, index) => 1016 - Math.min(index, 20) * 0.18),
  relativeHumidity2m: wave(70, -18),
  cloudCover: hours.map((_, index) => index >= 12 && index <= 28 ? 90 : 35),
  windU10: hours.map(() => 5),
  windV10: hours.map(() => -3),
  convectionProbability: hours.map((_, index) => index >= 18 && index <= 21 ? 82 : 12),
  stormConfidence: hours.map((_, index) => index >= 18 && index <= 21 ? 88 : 72),
  stormContradiction: hours.map((_, index) => index >= 18 && index <= 21 ? 8 : 16),
  windGust10: hours.map(() => 11),
  capeMl: hours.map((_, index) => index >= 18 && index <= 21 ? 1450 : 150),
  cinMl: hours.map((_, index) => index >= 18 && index <= 21 ? -30 : -120),
  visibility: hours.map(() => 10000),
  fogProbability: zeros(),
  freezingRainRisk: zeros(),
  foehnIndex: zeros(),
  frontDistanceKm: hours.map((_, index) => Math.abs(index - 19) * 18)
};

const storm = buildPointBulletin(stormSeries, "Palermo");
assert.equal(storm.title, "Bollettino per Palermo");
assert.equal(storm.confidence, "alta");
assert.equal(storm.dataCoverage, "alta");
assert.equal(storm.confidenceSemantics, "data-completeness-only-not-forecast-skill");
assert.equal(storm.completeness, 100);
assert.ok(storm.paragraphs.some(text => text.includes("Evoluzione giorno per giorno")));
assert.ok(storm.paragraphs.some(text => text.includes("15.0 mm")));
assert.ok(storm.paragraphs.some(text => text.includes("82/100")));
assert.ok(storm.paragraphs.some(text => text.includes("coerenza e copertura interne")));
assert.ok(storm.paragraphs.some(text => text.includes("non è calibrato")));
assert.ok(storm.paragraphs.some(text => text.includes("1450 J/kg")));
assert.ok(storm.paragraphs.some(text => text.includes("linea frontale oggettiva")));
assert.ok(storm.sections.some(section => section.title === "Temporali"));

const drySeries = {
  times: hours,
  temperature2m: wave(18, 4),
  rainStep: zeros(),
  cloudCover: hours.map(() => 15),
  convectionProbability: hours.map(() => null)
};
const dry = buildPointBulletin(drySeries, "Siracusa");
assert.equal(dry.confidence, "limitata");
assert.equal(dry.dataCoverage, "limitata");
assert.ok(dry.paragraphs.some(text => text.includes("Non emergono accumuli")));
assert.ok(!dry.paragraphs.some(text => text.includes("0%")));
assert.ok(!dry.paragraphs.join(" ").includes("NaN"));

console.log("Point bulletin tests passed");
