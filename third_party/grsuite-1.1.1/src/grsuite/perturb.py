"""Ensembles de forcages perturbes pour l'assimilation de donnees.

Traduction de airGRdatassim 0.1.4 R/CreateInputsPert.R (GPL-2, INRAE).
Les series de precipitation et/ou d'evapotranspiration potentielle sont
perturbees par un bruit multiplicatif issu d'un modele d'erreur
autoregressif d'ordre 1 (decorrelation de 1 jour pour la pluie, 2 jours
pour l'ETP, erreur fractionnaire de 0.65).

L'ordre des tirages aleatoires suit celui du code R (bruit blanc W puis,
au premier pas de temps, l'initialisation S0) afin que les tests puissent
rejouer les tirages exportes par tools/oracle/export_da_fixtures.R.

References
----------
Piazzi, G., Thirel, G., Perrin, C., Delaigue, O. (2021). Sequential data
assimilation for streamflow forecasting: assessing the sensitivity to
uncertainties and updated variables of a conceptual hydrological model at
basin scale. Water Resources Research 57, e2020WR028390.
doi:10.1029/2020WR028390
Piazzi, G., Delaigue, O. (2025). airGRdatassim: Ensemble-Based Data
Assimilation with GR Hydrological Models. R package version 0.1.4,
INRAE, HYCAR Research Unit. doi:10.32614/CRAN.package.airGRdatassim
"""

import math

import numpy as np

from ._rng import _NumpyRNG
from .models import data_alti_extrapolation_valery

__all__ = ["InputsPert", "DA_MODELS"]

#: Modeles acceptes par l'assimilation (journaliers, avec ou sans neige),
#: comme dans airGRdatassim.
DA_MODELS = ("GR4J", "GR5J", "GR6J",
             "CemaNeigeGR4J", "CemaNeigeGR5J", "CemaNeigeGR6J")

#: Decorrelation temporelle des erreurs [jour] (Tao dans le code R)
_TAO = {"Precip": 1.0, "PotEvap": 2.0}
#: Erreur fractionnaire (Eps dans le code R)
_EPS = {"Precip": 0.65, "PotEvap": 0.65}

_SQRT2 = math.sqrt(2.0)
_erfc = np.vectorize(math.erfc)


class InputsPert:
    """Ensembles perturbes de forcages meteo (CreateInputsPert).

    Parameters
    ----------
    model : str
        "GR4J", "GR5J", "GR6J" ou une variante CemaNeige.
    dates : array of numpy.datetime64
        Pas de temps, continus et tries.
    precip, pot_evap : array or None
        Series a perturber [mm/pas de temps] ; au moins une des deux est
        requise. Une serie laissee a None n'est pas perturbee : elle sera
        repetee a l'identique pour chaque membre par :func:`run_model_da`.
    temp_mean : array or None
        Temperature moyenne de l'air [degC], requise pour CemaNeige.
    z_inputs : float or None
        Altitude a laquelle les series se referent [m]. Defaut : mediane
        de ``hypso_data``.
    hypso_data : array or None
        Quantiles d'altitude du bassin [m], requis pour CemaNeige.
    n_layers : int
        Nombre de couches d'altitude (CemaNeige).
    nb_mbr : int
        Nombre de membres de l'ensemble (>= 2).
    seed : int or None
        Graine aleatoire ; comme dans le code R, le generateur est
        reinitialise a chaque pas de temps avec ``seed + i`` (i base 1).
    perturb_layers : bool
        CemaNeige seulement. Dans airGRdatassim 0.1.4, les series par
        couche d'altitude sont calculees sur la pluie non perturbee, si
        bien que la perturbation de la pluie n'atteint jamais le module de
        neige (seule l'ETP agit) ; c'est le comportement par defaut, fidele
        a la reference. Avec True, l'extrapolation altitudinale est refaite
        membre par membre (`layer_precip_ens`) : extension GRsuite, sans
        reference R.

    Attributes
    ----------
    precip, pot_evap : array (nb_time, nb_mbr) or None
        Ensembles perturbes ; None pour une variable non fournie.
    layer_precip_ens : list of array (nb_time, nb_mbr) or None
        Avec ``perturb_layers=True`` : la pluie de chaque membre extrapolee
        par couche d'altitude, utilisee par :func:`run_model_da` a la place
        de `layer_precip` ; None sinon.
    nb_mbr, model, dates, temp_mean, layer_precip, layer_temp_mean,
    layer_frac_solid_precip, z_layers
        Champs repris de l'InputsModel sous-jacent (non perturbes).
    """

    def __init__(self, model, dates, precip=None, pot_evap=None,
                 temp_mean=None, z_inputs=None, hypso_data=None, n_layers=5,
                 nb_mbr=50, seed=None, perturb_layers=False, _rng=None):
        if model not in DA_MODELS:
            raise ValueError(
                "modele non pris en charge pour l'assimilation : %s "
                "(attendu : %s)" % (model, ", ".join(DA_MODELS)))
        if not (isinstance(nb_mbr, (int, np.integer)) and nb_mbr >= 2):
            raise ValueError("'nb_mbr' doit etre un entier >= 2")
        if precip is None and pot_evap is None:
            raise ValueError("'precip' et 'pot_evap' sont tous les deux "
                             "absents : fournir au moins l'un des deux")

        self.model = model
        self.dates = np.asarray(dates)
        nb_time = self.dates.shape[0]
        self.nb_mbr = int(nb_mbr)

        # variables a perturber (MeteoNames dans le code R)
        meteo_names = []
        base = {}
        if precip is not None:
            meteo_names.append("Precip")
            base["Precip"] = np.asarray(precip, dtype=float)
        if pot_evap is not None:
            meteo_names.append("PotEvap")
            base["PotEvap"] = np.asarray(pot_evap, dtype=float)
        for name, arr in base.items():
            if arr.shape[0] != nb_time:
                raise ValueError("'%s' et 'dates' doivent avoir la meme "
                                 "longueur" % name.lower())

        # champs repris de l'InputsModel (non perturbes)
        self.temp_mean = (None if temp_mean is None
                          else np.asarray(temp_mean, dtype=float))
        if model.startswith("CemaNeige"):
            if self.temp_mean is None or hypso_data is None:
                raise ValueError("%s requiert 'temp_mean' et 'hypso_data'"
                                 % model)
            if z_inputs is None:
                z_inputs = float(np.median(np.asarray(hypso_data, float)))
            alti = data_alti_extrapolation_valery(
                self.dates, base.get("Precip", np.zeros(nb_time)),
                self.temp_mean, z_inputs=z_inputs, hypso_data=hypso_data,
                n_layers=n_layers)
            self.layer_precip = alti["LayerPrecip"]
            self.layer_temp_mean = alti["LayerTempMean"]
            self.layer_frac_solid_precip = alti["LayerFracSolidPrecip"]
            self.z_layers = alti["ZLayers"]
        else:
            self.layer_precip = None
            self.layer_temp_mean = None
            self.layer_frac_solid_precip = None
            self.z_layers = None

        # parametres du modele d'erreur AR(1), dans l'ordre de MeteoNames
        alfa = np.array([1.0 - 1.0 / _TAO[n] for n in meteo_names])
        eps = np.array([_EPS[n] for n in meteo_names])
        nb_meteo = len(meteo_names)

        rng = _rng if _rng is not None else _NumpyRNG(seed)

        # ensembles de reference : chaque colonne est la serie de base
        meteo_ens = np.empty((nb_meteo, self.nb_mbr, nb_time))
        for k, name in enumerate(meteo_names):
            meteo_ens[k] = np.tile(base[name], (self.nb_mbr, 1))

        s = None
        for t in range(nb_time):
            if seed is not None:
                rng.set_seed(seed + t + 1)   # iTime est base 1 dans le code R
            w = rng.rnorm(nb_meteo * self.nb_mbr, 0.0, 1.0)
            w = w.reshape(nb_meteo, self.nb_mbr)          # byrow = TRUE
            if t == 0:
                s0 = rng.runif(nb_meteo * self.nb_mbr, 0.0, 1.0)
                s0 = s0.reshape(nb_meteo, self.nb_mbr)    # byrow = TRUE
                s = (alfa[:, None] * s0
                     + np.sqrt(1.0 - alfa**2)[:, None] * w)
            else:
                s = (alfa[:, None] * s
                     + np.sqrt(1.0 - alfa**2)[:, None] * w)
            u = 0.5 * _erfc(s / _SQRT2)
            fi = (1.0 - eps)[:, None] + (2.0 * u) * eps[:, None]
            meteo_ens[:, :, t] = meteo_ens[:, :, t] * fi

        self.precip = (None if "Precip" not in meteo_names else
                       meteo_ens[meteo_names.index("Precip")].T.copy())
        self.pot_evap = (None if "PotEvap" not in meteo_names else
                         meteo_ens[meteo_names.index("PotEvap")].T.copy())

        # CemaNeige, extension optionnelle : l'extrapolation altitudinale
        # est refaite membre par membre sur la pluie perturbee ; la fraction
        # solide et la temperature par couche n'en dependent pas
        self.perturb_layers = bool(perturb_layers)
        self.layer_precip_ens = None
        if (model.startswith("CemaNeige") and self.precip is not None
                and self.perturb_layers):
            n_lay = len(self.layer_precip)
            ens = [np.empty((nb_time, self.nb_mbr)) for _ in range(n_lay)]
            for m in range(self.nb_mbr):
                alti_m = data_alti_extrapolation_valery(
                    self.dates, self.precip[:, m], self.temp_mean,
                    z_inputs=z_inputs, hypso_data=hypso_data,
                    n_layers=n_layers)
                for i in range(n_lay):
                    ens[i][:, m] = alti_m["LayerPrecip"][i]
            self.layer_precip_ens = ens

    def __len__(self):
        return self.dates.shape[0]
