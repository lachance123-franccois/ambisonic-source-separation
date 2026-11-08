import argparse
import os
from pathlib import Path

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import pandas as pd
import torch
from torch.utils.data import DataLoader

from beamformer import cardioid_beamformer
from config import TEST_DIR, VAL_DIR, CKPT_DIR, RESULTS_DIR
from metrics import si_sdr
from models import MaskAutoencoder, extract_voice
from plot_results import make_all
from scenes import SceneDataset, list_audio_files


def load_model(path):
    ckpt = torch.load(path, map_location="cpu")
    model = MaskAutoencoder(**ckpt["cfg"])
    model.load_state_dict(ckpt["state"])
    model.eval()
    return model, ckpt["mode"]


def resolve_test_dir(path):
    path = Path(path)
    if path.exists() and list_audio_files(path):
        print(f"Jeu de test : {path}")
        return path
    print(f"\n[AVERTISSEMENT] Jeu de test introuvable ou vide : {path}\n"
          f"                On se rabat sur le dossier de validation : {VAL_DIR}\n"
          f"                Les locuteurs de test sont alors ceux qui ont servi à choisir\n"
          f"                les checkpoints : résultats légèrement optimistes.\n")
    return VAL_DIR


def mirror(foa, az):
    #Scène symétrique gauche / droite : Y -> -Y et azimut -> -azimut
   # Moyenner une scène et son miroir supprime l'asymétrie propre au jeu de scènes
    foa_m = foa.clone()
    foa_m[:, 2] = -foa_m[:, 2]
    return foa_m, -az


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", default=str(TEST_DIR))
    p.add_argument("--model_dir", default=str(CKPT_DIR))
    p.add_argument("--n_items", type=int, default=200)
    p.add_argument("--static", action="store_true",
                   help="évalue les modèles conditionnés sur des scènes immobiles (par défaut : mobiles)")
    p.add_argument("--mirror", action="store_true",
                   help="évalue aussi chaque scène en miroir gauche / droite (temps d'évaluation x2)")
    p.add_argument("--az_errors", type=float, nargs="+",
                   default=[-30.0, -10.0, -5.0, 0.0, 2.0, 5.0, 10.0, 15.0, 20.0, 30.0])
    p.add_argument("--out_scenes", default=str(RESULTS_DIR / "eval_par_scene.csv"))
    p.add_argument("--seed", type=int, default=999)
    args = p.parse_args()

    files = list_audio_files(resolve_test_dir(args.data_dir))
    model_paths = sorted(Path(args.model_dir).glob("*.pt"))
    assert model_paths, f"Aucun fichier .pt trouvé dans {args.model_dir}"
    versions = [False, True] if args.mirror else [False]

    rows = []
    for path in model_paths:
        nom = path.stem
        model, mode = load_model(path)
        # Un modèle fixed est toujours testé dans ses conditions : immobile, cible à 0°.
        is_moving = False if mode == "fixed" else not args.static
        fixed_az = 0.0 if mode == "fixed" else None
        print(f"\n{'='*50}\nÉVALUATION : {nom}  (mode {mode}, scènes {'mobiles' if is_moving else 'immobiles'})\n{'='*50}")

        ds = SceneDataset(files, args.n_items, seed=args.seed, moving=is_moving, fixed_target_az=fixed_az)
        loader = DataLoader(ds, batch_size=16)

        for err in args.az_errors:
            res = {k: [] for k in ("inp", "bf", "hyp", "ml", "mir")}
            with torch.no_grad():
                for batch in loader:
                    for is_mirror in versions:
                        foa, az = batch["foa"], batch["az"]
                        if is_mirror:
                            foa, az = mirror(foa, az)
                        target = batch["target"]
                        az_asked = az + err
                        res["inp"].append(si_sdr(foa[:, 0], target))
                        res["bf"].append(si_sdr(cardioid_beamformer(foa, az_asked, p=0.5), target))
                        res["hyp"].append(si_sdr(cardioid_beamformer(foa, az_asked, p=0.25), target))
                        res["ml"].append(si_sdr(extract_voice(model, foa, az_asked), target))
                        res["mir"].append(torch.full((foa.shape[0],), is_mirror))
            # numéro de scène : l'ordre des lots est le même pour chaque version
            scene_ids = [i for b in range(len(loader)) for v in versions
                         for i in range(b * 16, min((b + 1) * 16, len(ds)))]
            inp, bf, hyp, ml, mir = (torch.cat(res[k]).numpy() for k in ("inp", "bf", "hyp", "ml", "mir"))
            for j, sc in enumerate(scene_ids):
                rows.append({"model_name": nom, "mode": mode, "moving": is_moving, "mirror": bool(mir[j]),
                             "az_error_deg": err, "scene": sc,
                             "sisdr_input": inp[j], "sisdr_beamformer": bf[j],
                             "sisdr_hypercardioid": hyp[j], "sisdr_model": ml[j]})
            print(f"Erreur {err:5.1f}° | modèle : moyenne {ml.mean():5.2f} dB, médiane {pd.Series(ml).median():5.2f} dB "
                  f"| cardioïde : moyenne {bf.mean():5.2f}, médiane {pd.Series(bf).median():5.2f} "
                  f"| modèle meilleur sur {100 * (ml > bf).mean():3.0f} % des scènes")

    Path(args.out_scenes).parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.out_scenes, index=False)
    print(f"\nSI-SDR par scène : {args.out_scenes}")

    # Résumé (moyenne, médiane, moyenne plafonnée, % gagné, IC), tolérances, asymétrie et figures
    make_all(args.out_scenes, Path(args.out_scenes).parent)


if __name__ == "__main__":
    main()