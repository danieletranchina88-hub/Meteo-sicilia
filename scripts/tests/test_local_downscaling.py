import copy
from pathlib import Path
import sys
import unittest
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.build_local_downscaling import build, sample, epoch, estimate

NOW=epoch('2026-09-05T01:00:00Z')
def fixtures():
    sites=[dict(id=f'S{i}',lat=38+i*.015,lon=13+i*.01,elevationM=100) for i in range(6)]
    forecast=dict(runTime='2026-09-05T00:00:00Z',stations=sites,
        times=[dict(validTime=f'2026-09-05T0{i}:00:00Z') for i in range(3)],
        fields=dict(temperature2m=dict(valuesByStation=[[20,22,24] for s in sites]),
                    terrainHeight=dict(valuesByStation=[[100,100,100] for s in sites])))
    obs=dict(source='test',stations=[dict(s,obsTime=NOW,tempC=24) for s in sites])
    return forecast,obs

class Tests(unittest.TestCase):
    def test_same_time_interpolation(self):
        f,o=fixtures()
        self.assertEqual(sample(f,'temperature2m',0,NOW-1800),21)
        self.assertIsNone(sample(f,'temperature2m',0,NOW-7200))
        self.assertIsNone(sample(f,'temperature2m',0,NOW+7200))
    def test_null_not_zero(self):
        f,o=fixtures();o['stations'][0]['tempC']=None
        p=build(f,o,NOW);self.assertEqual(len(p['stations']),5)
        f['fields']['temperature2m']['valuesByStation'][1][1]=None
        self.assertIsNone(sample(f,'temperature2m',1,NOW))
    def test_old_future_and_duplicate(self):
        f,o=fixtures();o['stations'][0]['obsTime']=NOW+1
        o['stations'][1]['obsTime']=NOW-7201
        o['stations'].append(copy.deepcopy(o['stations'][2]))
        p=build(f,o,NOW);self.assertEqual(len(p['stations']),4)
        self.assertEqual(p['status'],'withheld')
    def test_spatial_improvement_is_not_temporal_validation(self):
        f,o=fixtures();p=build(f,o,NOW)
        self.assertEqual(p['status'],'active-experimental')
        self.assertFalse(p['verification']['independentTemporalValidation'])
        self.assertLess(p['verification']['correctedMaeC'],p['verification']['baselineMaeC'])
    def test_holdout_never_uses_itself(self):
        f,o=fixtures();p=build(f,o,NOW);stations=p['stations']
        before=estimate(stations[0],stations,exclude=stations[0]['id'])
        stations[0]['residualC']=1000
        self.assertEqual(before,estimate(stations[0],stations,exclude=stations[0]['id']))
    def test_altitude_units_and_outlier(self):
        f,o=fixtures();o['stations'][0]['elevationM']=200
        o['stations'][0]['tempC']=21.35
        p=build(f,o,NOW);self.assertAlmostEqual(p['stations'][0]['residualC'],0)
        self.assertAlmostEqual(p['stations'][0]['altitudeCorrectionC'],-.65)
        o['stations'][1]['tempC']=45
        o['stations'][2]['elevationM']=1000
        p=build(f,o,NOW)
        self.assertEqual(p['rejected']['large-residual'],1)
        self.assertEqual(p['rejected']['altitude-gap'],1)
    def test_sparse_network_is_withheld(self):
        f,o=fixtures();o['stations']=o['stations'][:2]
        self.assertEqual(build(f,o,NOW)['status'],'withheld')
    def test_bad_time_axis(self):
        f,o=fixtures();f['times'].reverse()
        self.assertIsNone(sample(f,'temperature2m',0,NOW))

if __name__=='__main__': unittest.main()
