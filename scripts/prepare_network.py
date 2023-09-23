# -*- coding: utf-8 -*-
# SPDX-FileCopyrightText: : 2017-2023 The PyPSA-Eur Authors
#
# SPDX-License-Identifier: MIT

# coding: utf-8
"""
Prepare PyPSA network for solving according to :ref:`opts` and :ref:`ll`, such
as.

- adding an annual **limit** of carbon-dioxide emissions,
- adding an exogenous **price** per tonne emissions of carbon-dioxide (or other kinds),
- setting an **N-1 security margin** factor for transmission line capacities,
- specifying an expansion limit on the **cost** of transmission expansion,
- specifying an expansion limit on the **volume** of transmission expansion, and
- reducing the **temporal** resolution by averaging over multiple hours
  or segmenting time series into chunks of varying lengths using ``tsam``.

Relevant Settings
-----------------

.. code:: yaml

    costs:
        year:
        version:
        fill_values:
        emission_prices:
        marginal_cost:
        capital_cost:

    electricity:
        co2limit:
        max_hours:

.. seealso::
    Documentation of the configuration file ``config/config.yaml`` at
    :ref:`costs_cf`, :ref:`electricity_cf`

Inputs
------

- ``resources/costs.csv``: The database of cost assumptions for all included technologies for specific years from various sources; e.g. discount rate, lifetime, investment (CAPEX), fixed operation and maintenance (FOM), variable operation and maintenance (VOM), fuel costs, efficiency, carbon-dioxide intensity.
- ``networks/elec_s{simpl}_{clusters}.nc``: confer :ref:`cluster`

Outputs
-------

- ``networks/elec_s{simpl}_{clusters}_ec_l{ll}_{opts}.nc``: Complete PyPSA network that will be handed to the ``solve_network`` rule.

Description
-----------

.. tip::
    The rule :mod:`prepare_elec_networks` runs
    for all ``scenario`` s in the configuration file
    the rule :mod:`prepare_network`.
"""

import logging
import re

import numpy as np
import pandas as pd
import pypsa
from _helpers import configure_logging, get_opt
from add_electricity import load_costs, update_transmission_costs
from pypsa.descriptors import expand_series

idx = pd.IndexSlice

logger = logging.getLogger(__name__)


def find_opt(opts, expr):
    """
    Return if available the float after the expression.
    """
    for o in opts:
        if expr in o:
            m = re.findall("[0-9]*\.?[0-9]+$", o)
            if len(m) > 0:
                return True, float(m[0])
            else:
                return True, None
    return False, None


def add_co2limit(n, co2limit, Nyears=1.0):
    n.add(
        "GlobalConstraint",
        "CO2Limit",
        carrier_attribute="co2_emissions",
        sense="<=",
        constant=co2limit * Nyears,
    )


def add_gaslimit(n, gaslimit, Nyears=1.0):
    sel = n.carriers.index.intersection(["OCGT", "CCGT", "CHP"])
    n.carriers.loc[sel, "gas_usage"] = 1.0

    n.add(
        "GlobalConstraint",
        "GasLimit",
        carrier_attribute="gas_usage",
        sense="<=",
        constant=gaslimit * Nyears,
    )


def add_emission_prices(n, emission_prices={"co2": 0.0}, exclude_co2=False):
    if exclude_co2:
        emission_prices.pop("co2")
    ep = (
        pd.Series(emission_prices).rename(lambda x: x + "_emissions")
        * n.carriers.filter(like="_emissions")
    ).sum(axis=1)
    gen_ep = n.generators.carrier.map(ep) / n.generators.efficiency
    n.generators["marginal_cost"] += gen_ep
    n.generators_t["marginal_cost"] += gen_ep[n.generators_t["marginal_cost"].columns]
    su_ep = n.storage_units.carrier.map(ep) / n.storage_units.efficiency_dispatch
    n.storage_units["marginal_cost"] += su_ep


def add_dynamic_emission_prices(n):
    co2_price = pd.read_csv(snakemake.input.co2_price, index_col=0, parse_dates=True)
    co2_price = co2_price[~co2_price.index.duplicated()]
    co2_price = (
        co2_price.reindex(n.snapshots).fillna(method="ffill").fillna(method="bfill")
    )

    emissions = (
        n.generators.carrier.map(n.carriers.co2_emissions) / n.generators.efficiency
    )
    co2_cost = expand_series(emissions, n.snapshots).T.mul(co2_price.iloc[:, 0], axis=0)

    static = n.generators.marginal_cost
    dynamic = n.get_switchable_as_dense("Generator", "marginal_cost")

    marginal_cost = dynamic + co2_cost.reindex(columns=dynamic.columns, fill_value=0)
    n.generators_t.marginal_cost = marginal_cost.loc[:, marginal_cost.ne(static).any()]


def set_line_s_max_pu(n, s_max_pu=0.7):
    n.lines["s_max_pu"] = s_max_pu
    logger.info(f"N-1 security margin of lines set to {s_max_pu}")


def set_transmission_limit(n, ll_type, factor, costs, Nyears=1):
    links_dc_b = n.links.carrier == "DC" if not n.links.empty else pd.Series()

    _lines_s_nom = (
        np.sqrt(3)
        * n.lines.type.map(n.line_types.i_nom)
        * n.lines.num_parallel
        * n.lines.bus0.map(n.buses.v_nom)
    )
    lines_s_nom = n.lines.s_nom.where(n.lines.type == "", _lines_s_nom)

    col = "capital_cost" if ll_type == "c" else "length"
    ref = (
        lines_s_nom @ n.lines[col]
        + n.links.loc[links_dc_b, "p_nom"] @ n.links.loc[links_dc_b, col]
    )

    update_transmission_costs(n, costs)

    if factor == "opt" or float(factor) > 1.0:
        n.lines["s_nom_min"] = lines_s_nom
        n.lines["s_nom_extendable"] = True

        n.links.loc[links_dc_b, "p_nom_min"] = n.links.loc[links_dc_b, "p_nom"]
        n.links.loc[links_dc_b, "p_nom_extendable"] = True

    if factor != "opt":
        con_type = "expansion_cost" if ll_type == "c" else "volume_expansion"
        rhs = float(factor) * ref
        n.add(
            "GlobalConstraint",
            f"l{ll_type}_limit",
            type=f"transmission_{con_type}_limit",
            sense="<=",
            constant=rhs,
            carrier_attribute="AC, DC",
        )

    return n


def average_every_nhours(n, offset):
    logger.info(f"Resampling the network to {offset}")
    m = n.copy(with_time=False)

    snapshot_weightings = n.snapshot_weightings.resample(offset).sum()
    m.set_snapshots(snapshot_weightings.index)
    m.snapshot_weightings = snapshot_weightings

    for c in n.iterate_components():
        pnl = getattr(m, c.list_name + "_t")
        for k, df in c.pnl.items():
            if not df.empty:
                pnl[k] = df.resample(offset).mean()

    return m


def apply_time_segmentation(n, segments, solver_name="cbc"):
    logger.info(f"Aggregating time series to {segments} segments.")
    try:
        import tsam.timeseriesaggregation as tsam
    except:
        raise ModuleNotFoundError(
            "Optional dependency 'tsam' not found." "Install via 'pip install tsam'"
        )

    p_max_pu_norm = n.generators_t.p_max_pu.max()
    p_max_pu = n.generators_t.p_max_pu / p_max_pu_norm

    load_norm = n.loads_t.p_set.max()
    load = n.loads_t.p_set / load_norm

    inflow_norm = n.storage_units_t.inflow.max()
    inflow = n.storage_units_t.inflow / inflow_norm

    raw = pd.concat([p_max_pu, load, inflow], axis=1, sort=False)

    agg = tsam.TimeSeriesAggregation(
        raw,
        hoursPerPeriod=len(raw),
        noTypicalPeriods=1,
        noSegments=int(segments),
        segmentation=True,
        solver=solver_name,
    )

    segmented = agg.createTypicalPeriods()

    weightings = segmented.index.get_level_values("Segment Duration")
    offsets = np.insert(np.cumsum(weightings[:-1]), 0, 0)
    snapshots = [n.snapshots[0] + pd.Timedelta(f"{offset}h") for offset in offsets]

    n.set_snapshots(pd.DatetimeIndex(snapshots, name="name"))
    n.snapshot_weightings = pd.Series(
        weightings, index=snapshots, name="weightings", dtype="float64"
    )

    segmented.index = snapshots
    n.generators_t.p_max_pu = segmented[n.generators_t.p_max_pu.columns] * p_max_pu_norm
    n.loads_t.p_set = segmented[n.loads_t.p_set.columns] * load_norm
    n.storage_units_t.inflow = segmented[n.storage_units_t.inflow.columns] * inflow_norm

    return n


def enforce_autarky(n, only_crossborder=False):
    if only_crossborder:
        lines_rm = n.lines.loc[
            n.lines.bus0.map(n.buses.country) != n.lines.bus1.map(n.buses.country)
        ].index
        links_rm = n.links.loc[
            n.links.bus0.map(n.buses.country) != n.links.bus1.map(n.buses.country)
        ].index
    else:
        lines_rm = n.lines.index
        links_rm = n.links.loc[n.links.carrier == "DC"].index
    n.mremove("Line", lines_rm)
    n.mremove("Link", links_rm)


def set_line_nom_max(
    n,
    s_nom_max_set=np.inf,
    p_nom_max_set=np.inf,
    s_nom_max_ext=np.inf,
    p_nom_max_ext=np.inf,
):
    if np.isfinite(s_nom_max_ext) and s_nom_max_ext > 0:
        logger.info(f"Limiting line extensions to {s_nom_max_ext} MW")
        n.lines["s_nom_max"] = n.lines["s_nom"] + s_nom_max_ext

    if np.isfinite(p_nom_max_ext) and p_nom_max_ext > 0:
        logger.info(f"Limiting line extensions to {p_nom_max_ext} MW")
        hvdc = n.links.index[n.links.carrier == "DC"]
        n.links.loc[hvdc, "p_nom_max"] = n.links.loc[hvdc, "p_nom"] + p_nom_max_ext

    n.lines.s_nom_max.clip(upper=s_nom_max_set, inplace=True)
    n.links.p_nom_max.clip(upper=p_nom_max_set, inplace=True)


# %%
if __name__ == "__main__":
    if "snakemake" not in globals():
        from _helpers import mock_snakemake

        snakemake = mock_snakemake(
            "prepare_network", simpl="", clusters="37", ll="v1.0", opts="Ept"
        )
    configure_logging(snakemake)

    opts = snakemake.wildcards.opts.split("-")

    n = pypsa.Network(snakemake.input[0])
    Nyears = n.snapshot_weightings.objective.sum() / 8760.0
    costs = load_costs(
        snakemake.input.tech_costs,
        snakemake.params.costs,
        snakemake.params.max_hours,
        Nyears,
    )

    set_line_s_max_pu(n, snakemake.params.lines["s_max_pu"])

    # temporal averaging
    nhours_config = snakemake.params.snapshots.get("resolution", False)
    nhours_wildcard = get_opt(opts, r"^\d+h$")
    nhours = nhours_wildcard or nhours_config
    if nhours:
        n = average_every_nhours(n, nhours)

    # segments with package tsam
    time_seg_config = snakemake.params.snapshots.get("segmentation", False)
    time_seg_wildcard = get_opt(opts, r"^\d+seg$")
    time_seg = time_seg_wildcard or time_seg_config
    if time_seg:
        solver_name = snakemake.config["solving"]["solver"]["name"]
        n = apply_time_segmentation(n, time_seg, solver_name)

    Co2L_config = snakemake.params.co2limit_enable
    Co2L_wildcard, co2limit_wildcard = find_opt(opts, "Co2L")
    if Co2L_wildcard or Co2L_config:
        if co2limit_wildcard is not None:
            co2limit = co2limit_wildcard * snakemake.params.co2base
            add_co2limit(n, co2limit, Nyears)
            logger.info("Setting CO2 limit according to wildcard value.")
        else:
            add_co2limit(n, snakemake.params.co2limit, Nyears)
            logger.info("Setting CO2 limit according to config value.")

    CH4L_config = snakemake.params.gaslimit_enable
    CH4L_wildcard, gaslimit_wildcard = find_opt(opts, "CH4L")
    if CH4L_wildcard or CH4L_config:
        if gaslimit_wildcard is not None:
            gaslimit = gaslimit_wildcard * 1e6
            add_gaslimit(n, gaslimit, Nyears)
            logger.info("Setting gas usage limit according to wildcard value.")
        else:
            add_gaslimit(n, snakemake.params.gaslimit, Nyears)
            logger.info("Setting gas usage limit according to config value.")

    for o in opts:
        if "+" not in o:
            continue
        oo = o.split("+")
        suptechs = map(lambda c: c.split("-", 2)[0], n.carriers.index)
        if oo[0].startswith(tuple(suptechs)):
            carrier = oo[0]
            # handles only p_nom_max as stores and lines have no potentials
            attr_lookup = {"p": "p_nom_max", "c": "capital_cost", "m": "marginal_cost"}
            attr = attr_lookup[oo[1][0]]
            factor = float(oo[1][1:])
            if carrier == "AC":  # lines do not have carrier
                n.lines[attr] *= factor
            else:
                comps = {"Generator", "Link", "StorageUnit", "Store"}
                for c in n.iterate_components(comps):
                    sel = c.df.carrier.str.contains(carrier)
                    c.df.loc[sel, attr] *= factor

    emission_prices = snakemake.params.costs["emission_prices"]
    Ept_config = emission_prices.get("co2_monthly_prices", False)
    Ept_wildcard = "Ept" in opts
    Ep_config = emission_prices.get("enable", False)
    Ep_wildcard, co2_wildcard = find_opt(opts, "Ep")

    if Ept_wildcard or Ept_config:
        logger.info(
            "Setting time dependent emission prices according spot market price"
        )
        add_dynamic_emission_prices(n)
    elif Ep_wildcard or Ep_config:
        if co2_wildcard is not None:
            logger.info("Setting CO2 prices according to wildcard value.")
            add_emission_prices(n, dict(co2=co2_wildcard))
        else:
            logger.info("Setting CO2 prices according to config value.")
            add_emission_prices(
                n, dict(co2=snakemake.params.costs["emission_prices"]["co2"])
            )

    ll_type, factor = snakemake.wildcards.ll[0], snakemake.wildcards.ll[1:]
    set_transmission_limit(n, ll_type, factor, costs, Nyears)

    set_line_nom_max(
        n,
        s_nom_max_set=snakemake.params.lines.get("s_nom_max", np.inf),
        p_nom_max_set=snakemake.params.links.get("p_nom_max", np.inf),
        s_nom_max_ext=snakemake.params.lines.get("max_extension", np.inf),
        p_nom_max_ext=snakemake.params.links.get("max_extension", np.inf),
    )

    autarky_config = snakemake.params.autarky
    if "ATK" in opts or autarky_config.get("enable", False):
        only_crossborder = False
        if "ATKc" in opts or autarky_config.get("by_country", False):
            only_crossborder = True
        enforce_autarky(n, only_crossborder=only_crossborder)

    n.meta = dict(snakemake.config, **dict(wildcards=dict(snakemake.wildcards)))
    n.export_to_netcdf(snakemake.output[0])
