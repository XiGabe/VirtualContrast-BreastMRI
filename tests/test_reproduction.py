import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score, average_precision_score

from virtualcontrast.data import read_manifest
from virtualcontrast.evaluate import evaluate, load_predictions
from virtualcontrast.routing import batch_metrics, simulate

ROOT = Path(__file__).resolve().parents[1]


def test_reference_table(tmp_path):
    records = evaluate(ROOT/'reproducibility/predictions',tmp_path)
    expected = pd.read_csv(ROOT/'reproducibility/expected_table1.csv')
    actual = pd.DataFrame(records)
    pd.testing.assert_frame_equal(actual,expected,check_exact=False,rtol=0,atol=1e-12)


def test_tie_aware_metrics():
    y = np.array([1,0,1,0,0,1])
    scores = np.array([[0,0,0,0,0,0],[.5,.5,.8,.8,.1,.1],[1,2,3,4,5,6]])
    expected = [[roc_auc_score(y,z),average_precision_score(y,z)] for z in scores]
    np.testing.assert_allclose(batch_metrics(y,scores),expected,atol=1e-12)


def test_routing_endpoints_and_patient_policy():
    frames = load_predictions(ROOT/'reproducibility/predictions')
    rows = simulate(frames,permutations=7)
    expected = pd.read_csv(ROOT/'reproducibility/expected_routing.csv')
    for i in [0,60,100]:
        for metric in ['auroc','auprc']:
            assert rows[i]['confidence_'+metric] == pytest.approx(expected.iloc[i]['confidence_'+metric],abs=1e-12)
    for i in [0,100]:
        assert rows[i]['random_auroc'] == pytest.approx(rows[i]['confidence_auroc'],abs=1e-12)
    assert rows[60]['patients'] == 147


def test_frozen_cohort():
    rows = read_manifest(ROOT/'reproducibility/cohort.csv')
    assert len({r['patient'] for r in rows}) == 1410
    assert [sum(r['split']==s for r in rows) for s in ['train','val','test']] == [1888,375,473]


def test_leakage_rejected(tmp_path):
    rows = list(csv.DictReader((ROOT/'reproducibility/cohort.csv').open()))[:2]
    rows[1]['split'] = 'test' if rows[0]['split']=='train' else 'train'
    rows[1]['patient'] = rows[0]['patient']
    p=tmp_path/'bad.csv'
    with p.open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    with pytest.raises(ValueError,match='crosses splits'):
        read_manifest(p)


def test_misaligned_predictions_rejected(tmp_path):
    for seed in [2026,2027,2028]:
        d=pd.read_csv(ROOT/f'reproducibility/predictions/seed_{seed}.csv')
        if seed==2027:
            d.loc[0,'label']=1-d.loc[0,'label']
        d.to_csv(tmp_path/f'seed_{seed}.csv',index=False)
    with pytest.raises(AssertionError):
        load_predictions(tmp_path)
