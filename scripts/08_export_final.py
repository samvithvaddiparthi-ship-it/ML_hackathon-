"""Stage A5 - export the final inference artifacts used by the API (artifacts/final/):
  emission_model.joblib  the selected XGBoost (fit on all 80 train trips)
  hmm_config.json        decoder parameters chosen in Stage A4 (written by 07_hmm_experiments.py)
  model_card.json        name/version, features, hyper-parameters, training data, headline results
Then smoke-tests predict_trip + score_trip on every student-test trip.
"""
import sys, json, shutil, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
import pandas as pd
from config import RESULTS, ARTIFACTS, STUDENT_TEST_DIR
from pathsnap.features import FEATURES, RADIUS, MAX_K, SIGMA

FINAL = ARTIFACTS / "final"
SRC = {"XGBoost_untuned": ARTIFACTS / "models" / "XGBoost.joblib",
       "XGBoost_tuned": ARTIFACTS / "models" / "XGBoost_tuned.joblib"}


def main():
    hmm = json.load(open(FINAL / "hmm_config.json"))
    em = hmm["emission_model"]
    shutil.copyfile(SRC[em], FINAL / "emission_model.joblib")
    st = pd.read_csv(RESULTS / "hmm" / "student_test_summary.csv")
    sel = st[st.model == em].set_index("decoder")
    params = (json.load(open(RESULTS / "bayes_opt" / "best_params.json"))["params"] if em == "XGBoost_tuned"
              else json.load(open(RESULTS / "models" / "model_configs.json"))["XGBoost"]["config"])
    card = {
        "name": f"{em.replace('_', ' ')} + HMM/Viterbi", "version": "1.0",
        "emission_model": em, "emission_params": params, "features": FEATURES,
        "candidates": {"radius_m": RADIUS, "max_k": MAX_K, "sigma_m": SIGMA},
        "hmm": {k: hmm[k] for k in ("beta", "w_e", "gap_s", "vmax")},
        "trained_on": "80 TRAIN trips (labelled observations); decoder tuned on TRAIN out-of-fold emissions",
        "student_test": {d: {k: round(float(sel.loc[d, k]), 4) for k in ("top1_acc", "route_f1", "route_continuity")}
                         for d in sel.index},
    }
    json.dump(card, open(FINAL / "model_card.json", "w"), indent=1)
    print(json.dumps(card, indent=1))

    from pathsnap.pipeline import predict_trip
    from pathsnap.evaluate import score_trip
    rows = []
    for d in sorted(p for p in STUDENT_TEST_DIR.iterdir() if p.is_dir()):
        t0 = time.perf_counter(); pred = predict_trip(d); ms = (time.perf_counter() - t0) * 1e3
        m = score_trip(pred, d); m.update(trip_id=d.name, total_ms=round(ms), **{f"t_{k}": v for k, v in pred["timing_ms"].items()})
        rows.append(m)
    df = pd.DataFrame(rows).set_index("trip_id")
    df.to_csv(RESULTS / "hmm" / "pipeline_smoke_student_test.csv")
    print(df[["n_obs", "top1_accuracy_classifier_only", "top1_accuracy", "macro_f1", "route_f1",
              "route_continuity_classifier_only", "route_continuity", "total_ms"]].to_string())


if __name__ == "__main__":
    main()
