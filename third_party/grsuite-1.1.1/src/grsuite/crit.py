"""Criteres d'erreur (fonctions objectif).

Traduction de airGR/R/UtilsErrorCrit.R, ErrorCrit_NSE.R, ErrorCrit_KGE.R,
ErrorCrit_KGE2.R, ErrorCrit_RMSE.R, CreateInputsCrit.R (v1.7.9).

Convention airGR conservee : `multiplier` vaut -1 pour les criteres a
maximiser (NSE, KGE, KGE2) et +1 pour ceux a minimiser (RMSE), de sorte
que le calage minimise toujours `crit_value * multiplier`.

References
----------
Nash, J.E., Sutcliffe, J.V. (1970). River flow forecasting through
    conceptual models part I. Journal of Hydrology 10, 282-290.
    doi:10.1016/0022-1694(70)90255-6                              [NSE]
Gupta, H.V., Kling, H., Yilmaz, K.K., Martinez, G.F. (2009).
    Decomposition of the mean squared error and NSE performance
    criteria. Journal of Hydrology 377, 80-91.
    doi:10.1016/j.jhydrol.2009.08.003                             [KGE]
Kling, H., Fuchs, M., Paulin, M. (2012). Runoff conditions in the upper
    Danube basin under an ensemble of climate change scenarios.
    Journal of Hydrology 424-425, 264-277.
    doi:10.1016/j.jhydrol.2012.01.011                            [KGE2]
Box, G.E.P., Cox, D.R. (1964). An analysis of transformations. Journal
    of the Royal Statistical Society B 26, 211-243.            [boxcox]
Santos, L., Thirel, G., Perrin, C. (2018). Technical note: pitfalls in
    using log-transformed flows within the KGE criterion. Hydrology and
    Earth System Sciences 22, 4583-4591. doi:10.5194/hess-22-4583-2018
"""

import numpy as np

__all__ = [
    "InputsCrit",
    "OutputsCrit",
    "error_crit",
    "error_crit_nse",
    "error_crit_kge",
    "error_crit_kge2",
    "error_crit_rmse",
    "CRIT_FUNCS",
]


def _sd(x):
    """Ecart-type d'echantillon (denominateur n-1), comme stats::sd de R."""
    return np.std(x, ddof=1)


class InputsCrit:
    """Definit un critere : variable observee, periode, transformation.

    Parameters
    ----------
    fun_crit : str
        "NSE", "KGE", "KGE2" ou "RMSE".
    obs : array
        Serie observee sur la periode de simulation (IndPeriod_Run).
    transfo : str
        "", "sqrt", "log", "inv", "sort", "boxcox" ou "^p" (ex. "^0.5").
    epsilon : float or None
        Terme ajoute a obs et sim avant transformation.
    bool_crit : array of bool or None
        Pas de temps a prendre en compte.
    weights : float or None
        Poids, pour un critere composite uniquement.
    var_obs : str
        "Q" (defaut), "SCA" ou "SWE". Pour SCA et SWE, `obs` est comme
        dans CreateInputsCrit une liste d'un vecteur par couche d'altitude
        (moyennee ici, et comparee a la moyenne des couches simulees) ; un
        vecteur unique est accepte comme moyenne deja calculee.
    id_layer : sequence of int or None
        Couches (base 0) auxquelles se rapportent les vecteurs de `obs`
        pour SCA et SWE ; par defaut les premieres, dans l'ordre. None pour
        "Q".
    """

    def __init__(self, fun_crit, obs, transfo="", epsilon=None,
                 bool_crit=None, weights=None, var_obs="Q", id_layer=None):
        fun_crit = fun_crit.upper()
        if fun_crit not in CRIT_FUNCS:
            raise ValueError("critere inconnu : %s" % fun_crit)
        if var_obs not in ("Q", "SCA", "SWE"):
            raise ValueError("variable observee inconnue : %s" % var_obs)
        self.fun_crit = fun_crit
        self.id_layer = None
        if var_obs in ("SCA", "SWE") and _is_layer_list(obs):
            layers = [np.asarray(o, dtype=float) for o in obs]
            if id_layer is None:
                id_layer = list(range(len(layers)))
            if len(id_layer) != len(layers):
                raise ValueError("'id_layer' doit designer une couche par "
                                 "vecteur de 'obs' (%i)" % len(layers))
            self.obs = np.mean(layers, axis=0)   # rowMeans, CreateInputsCrit
        else:
            self.obs = np.asarray(obs, dtype=float)
        if id_layer is not None:
            self.id_layer = [int(i) for i in id_layer]
        if var_obs == "SCA":
            finite = self.obs[np.isfinite(self.obs)]
            if finite.size and (finite.min() < 0.0 or finite.max() > 1.0):
                raise ValueError("'obs' hors de [0, 1] pour \"SCA\"")
        self.transfo = transfo
        self.epsilon = epsilon
        self.var_obs = var_obs
        self.weights = weights
        if bool_crit is None:
            self.bool_crit = np.ones(self.obs.shape[0], dtype=bool)
        else:
            self.bool_crit = np.asarray(bool_crit, dtype=bool)
        if self.bool_crit.shape[0] != self.obs.shape[0]:
            raise ValueError("'bool_crit' et 'obs' doivent avoir la meme longueur")


def _is_layer_list(obs):
    """Vrai pour une liste de vecteurs (un par couche), faux pour un vecteur."""
    if isinstance(obs, np.ndarray):
        return obs.ndim == 2
    return (isinstance(obs, (list, tuple)) and len(obs) > 0
            and np.ndim(obs[0]) == 1)


class InputsCritCompo:
    """Critere composite : moyenne ponderee de plusieurs criteres simples.

    Comme CreateInputsCrit, refuse le RMSE, qui n'est pas adimensionnel.
    """

    def __init__(self, inputs_crit_list):
        self.items = list(inputs_crit_list)
        if any(ic.weights is None for ic in self.items):
            raise ValueError("chaque critere composite doit avoir un poids")
        if any(ic.fun_crit == "RMSE" for ic in self.items):
            raise ValueError("un critere composite ne peut pas inclure le "
                             "RMSE, qui n'est pas adimensionnel")


class OutputsCrit:
    """Resultat d'un critere (OutputsCrit d'airGR).

    Pour un critere composite, `sub_crit_values` et `sub_crit_names` sont
    les valeurs et noms des sous-criteres (MultiCritValues, MultiCritNames)
    et `weights` leurs poids normalises (MultiCritWeights) ; pour KGE et
    KGE2 ce sont les composantes r, alpha (ou gamma) et beta.
    """

    def __init__(self, crit_value, crit_name, crit_best_value, multiplier,
                 sub_crit_values=None, sub_crit_names=None,
                 ind_not_computed=None, weights=None):
        self.crit_value = crit_value
        self.crit_name = crit_name
        self.crit_best_value = crit_best_value
        self.multiplier = multiplier
        self.sub_crit_values = sub_crit_values
        self.sub_crit_names = sub_crit_names
        self.ind_not_computed = ind_not_computed
        self.weights = weights

    def __repr__(self):
        return "Crit. %s = %.4f" % (self.crit_name, self.crit_value)


def _layer_mean(outputs_model, name, id_layer=None):
    """Moyenne des couches CemaNeige d'une variable de neige.

    Equivalent de rowMeans(sapply(OutputsModel$CemaNeigeLayers[idLayer],
    "[[", name)) dans .ErrorCrit : les variables SCA (Gratio) et SWE
    (SnowPack) ne sont pas des sorties de bassin mais des sorties par
    couche.
    """
    layers = outputs_model.get("CemaNeigeLayers")
    if not layers:
        raise ValueError(
            "la variable observee '%s' requiert les sorties d'un modele "
            "CemaNeige (cle 'CemaNeigeLayers' absente)" % name)
    if id_layer is not None:
        if any(i < 0 or i >= len(layers) for i in id_layer):
            raise ValueError("'id_layer' hors des %i couches du modele"
                             % len(layers))
        layers = [layers[i] for i in id_layer]
    return np.mean([np.asarray(layer[name], dtype=float) for layer in layers],
                   axis=0)


def _prepare(inputs_crit, outputs_model, crit):
    """Equivalent de la fonction interne .ErrorCrit de airGR."""
    transfo = inputs_crit.transfo
    crit_var = inputs_crit.var_obs

    transfo_pow = None
    if transfo == "":
        crit_name = "%s[%s]" % (crit, crit_var)
    elif transfo in ("sqrt", "log", "sort", "boxcox"):
        crit_name = "%s[%s(%s)]" % (crit, transfo, crit_var)
    elif transfo == "inv":
        crit_name = "%s[1/%s]" % (crit, crit_var)
    elif "^" in transfo:
        transfo_pow = float(transfo.replace("^", ""))
        crit_name = "%s[%s^%g]" % (crit, crit_var, transfo_pow)
    else:
        raise ValueError("transformation inconnue : %s" % transfo)

    bool_crit = inputs_crit.bool_crit.copy()

    var_obs = inputs_crit.obs.astype(float).copy()
    var_obs[~bool_crit] = np.nan

    if crit_var == "Q":
        var_sim = np.asarray(outputs_model["Qsim"], dtype=float).copy()
    elif crit_var == "SCA":
        var_sim = _layer_mean(outputs_model, "Gratio", inputs_crit.id_layer)
    elif crit_var == "SWE":
        var_sim = _layer_mean(outputs_model, "SnowPack", inputs_crit.id_layer)
    else:
        raise ValueError("variable observee inconnue : %s" % crit_var)
    if var_obs.shape[0] != var_sim.shape[0]:
        raise ValueError("'obs' (%i valeurs) doit couvrir la periode de "
                         "simulation (%i pas de temps)"
                         % (var_obs.shape[0], var_sim.shape[0]))
    var_sim[~bool_crit] = np.nan

    eps = inputs_crit.epsilon
    if eps is not None and transfo != "boxcox":
        var_obs = var_obs + eps
        var_sim = var_sim + eps

    with np.errstate(divide="ignore", invalid="ignore"):
        if transfo == "sqrt":
            var_obs = np.sqrt(var_obs)
            var_sim = np.sqrt(var_sim)
        elif transfo == "log":
            var_obs = np.log(var_obs)
            var_sim = np.log(var_sim)
            var_sim[var_sim < -1e100] = np.nan
        elif transfo == "inv":
            var_obs = 1.0 / var_obs
            var_sim = 1.0 / var_sim
            var_sim[np.abs(var_sim) > 1e100] = np.nan
        elif transfo == "sort":
            var_sim[np.isnan(var_obs)] = np.nan
            var_sim = np.sort(var_sim)          # NaN en fin, comme na.last=TRUE
            var_obs = np.sort(var_obs)
            bool_crit = np.sort(bool_crit)[::-1]
        elif transfo == "boxcox":
            mu = (0.01 * np.nanmean(var_obs)) ** 0.25
            var_sim = (var_sim ** 0.25 - mu) / 0.25
            var_obs = (var_obs ** 0.25 - mu) / 0.25
        elif transfo_pow is not None:
            var_obs = var_obs ** transfo_pow
            var_sim = var_sim ** transfo_pow

    ts_ignore = ~np.isfinite(var_obs) | ~np.isfinite(var_sim) | ~bool_crit
    ind_ignore = np.where(ts_ignore)[0]
    n_used = int(np.sum(~ts_ignore))

    if n_used == 0 or (n_used == 1 and crit in ("KGE", "KGE2")):
        crit_compute = False
    else:
        crit_compute = True

    return {
        "var_obs": var_obs,
        "var_sim": var_sim,
        "crit_name": crit_name,
        "crit_var": crit_var,
        "crit_compute": crit_compute,
        "ts_ignore": ts_ignore,
        "ind_ignore": ind_ignore if ind_ignore.size else None,
    }


def error_crit_nse(inputs_crit, outputs_model):
    ec = _prepare(inputs_crit, outputs_model, "NSE")
    crit_value = np.nan
    if ec["crit_compute"]:
        ok = ~ec["ts_ignore"]
        o = ec["var_obs"][ok]
        s = ec["var_sim"][ok]
        emod = np.sum((s - o) ** 2)
        eref = np.sum((o - np.mean(o)) ** 2)
        if emod == 0.0 and eref == 0.0:
            value = 0.0
        else:
            value = 1.0 - emod / eref
        if np.isfinite(value):
            crit_value = value
    return OutputsCrit(crit_value, ec["crit_name"], 1.0, -1.0,
                       ind_not_computed=ec["ind_ignore"])


def error_crit_kge(inputs_crit, outputs_model):
    ec = _prepare(inputs_crit, outputs_model, "KGE")
    crit_value = np.nan
    sub = np.full(3, np.nan)
    names = ["r", "alpha", "beta"]
    if ec["crit_compute"]:
        ok = ~ec["ts_ignore"]
        o = ec["var_obs"][ok]
        s = ec["var_sim"][ok]
        mo = np.mean(o)
        ms = np.mean(s)

        numer = np.sum((o - mo) * (s - ms))
        deno1 = np.sqrt(np.sum((o - mo) ** 2))
        deno2 = np.sqrt(np.sum((s - ms) ** 2))
        if numer == 0.0:
            value = 1.0 if (deno1 == 0.0 and deno2 == 0.0) else 0.0
        else:
            value = numer / (deno1 * deno2)
        if np.isfinite(value):
            sub[0] = value

        numer = _sd(s)
        denom = _sd(o)
        value = 1.0 if (numer == 0.0 and denom == 0.0) else numer / denom
        if np.isfinite(value):
            sub[1] = value

        value = 1.0 if (ms == 0.0 and mo == 0.0) else ms / mo
        if np.isfinite(value):
            sub[2] = value

        if not np.any(np.isnan(sub)):
            crit_value = 1.0 - np.sqrt(
                (sub[0] - 1.0) ** 2 + (sub[1] - 1.0) ** 2 + (sub[2] - 1.0) ** 2
            )
    return OutputsCrit(crit_value, ec["crit_name"], 1.0, -1.0,
                       sub_crit_values=sub, sub_crit_names=names,
                       ind_not_computed=ec["ind_ignore"])


def error_crit_kge2(inputs_crit, outputs_model):
    ec = _prepare(inputs_crit, outputs_model, "KGE2")
    crit_value = np.nan
    sub = np.full(3, np.nan)
    names = ["r", "gamma", "beta"]
    if ec["crit_compute"]:
        ok = ~ec["ts_ignore"]
        o = ec["var_obs"][ok]
        s = ec["var_sim"][ok]
        mo = np.mean(o)
        ms = np.mean(s)

        numer = np.sum((o - mo) * (s - ms))
        deno1 = np.sqrt(np.sum((o - mo) ** 2))
        deno2 = np.sqrt(np.sum((s - ms) ** 2))
        if numer == 0.0:
            value = 1.0 if (deno1 == 0.0 and deno2 == 0.0) else 0.0
        else:
            value = numer / (deno1 * deno2)
        if np.isfinite(value):
            sub[0] = value

        if ms == 0.0:
            cv_sim = 1.0 if _sd(s) == 0.0 else 99999.0
        else:
            cv_sim = _sd(s) / ms
        if mo == 0.0:
            cv_obs = 1.0 if _sd(o) == 0.0 else 99999.0
        else:
            cv_obs = _sd(o) / mo
        value = 1.0 if (cv_sim == 0.0 and cv_obs == 0.0) else cv_sim / cv_obs
        if np.isfinite(value):
            sub[1] = value

        value = 1.0 if (ms == 0.0 and mo == 0.0) else ms / mo
        if np.isfinite(value):
            sub[2] = value

        if not np.any(np.isnan(sub)):
            crit_value = 1.0 - np.sqrt(
                (sub[0] - 1.0) ** 2 + (sub[1] - 1.0) ** 2 + (sub[2] - 1.0) ** 2
            )
    return OutputsCrit(crit_value, ec["crit_name"], 1.0, -1.0,
                       sub_crit_values=sub, sub_crit_names=names,
                       ind_not_computed=ec["ind_ignore"])


def error_crit_rmse(inputs_crit, outputs_model):
    ec = _prepare(inputs_crit, outputs_model, "RMSE")
    crit_value = np.nan
    if ec["crit_compute"]:
        o = ec["var_obs"]
        s = ec["var_sim"]
        diff2 = (s - o) ** 2
        numer = np.nansum(diff2)
        denom = int(np.sum(~np.isnan(o) & ~np.isnan(s)))
        value = 0.0 if numer == 0.0 else np.sqrt(numer / denom)
        if np.isfinite(value):
            crit_value = value
    return OutputsCrit(crit_value, ec["crit_name"], 0.0, 1.0,
                       ind_not_computed=ec["ind_ignore"])


CRIT_FUNCS = {
    "NSE": error_crit_nse,
    "KGE": error_crit_kge,
    "KGE2": error_crit_kge2,
    "RMSE": error_crit_rmse,
}


def error_crit(inputs_crit, outputs_model):
    """Evalue un critere simple ou composite (ErrorCrit d'airGR).

    Un critere composite vaut la somme des sous-criteres ponderee par leurs
    poids normalises ; comme dans airGR, `crit_value` garde le signe naturel
    des sous-criteres (un composite de NSE vaut au mieux 1), `multiplier`
    est celui, commun, des sous-criteres et `crit_best_value` vaut 1.
    Melanger des criteres a maximiser et a minimiser est refuse.
    """
    if isinstance(inputs_crit, InputsCritCompo):
        outs = [CRIT_FUNCS[item.fun_crit](item, outputs_model)
                for item in inputs_crit.items]
        multipliers = [out.multiplier for out in outs]
        if any(m != multipliers[0] for m in multipliers):
            raise ValueError(
                "un critere composite ne peut pas melanger des criteres a "
                "maximiser (NSE, KGE, KGE2) et a minimiser (RMSE)")
        weights = np.array([float(item.weights) for item in inputs_crit.items])
        weights = weights / np.sum(weights)
        values = np.array([out.crit_value for out in outs], dtype=float)
        crit_value = float(np.sum(values * weights))   # NaN se propage
        ignored = [out.ind_not_computed for out in outs
                   if out.ind_not_computed is not None]
        ind_ignore = (np.unique(np.concatenate(ignored)) if ignored
                      else None)
        return OutputsCrit(crit_value, "Composite", 1.0, multipliers[0],
                           sub_crit_values=values,
                           sub_crit_names=[out.crit_name for out in outs],
                           ind_not_computed=ind_ignore, weights=weights)
    return CRIT_FUNCS[inputs_crit.fun_crit](inputs_crit, outputs_model)
