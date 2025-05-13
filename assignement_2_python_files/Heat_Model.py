from pyomo.environ import *

model = ConcreteModel()

# Zeitbereich: 24 Stunden
model.T = RangeSet(0, 23)

# Parameter (hier Platzhalter, später überschreibbar)
model.d = Param(model.T, initialize=lambda model, t: 0, mutable=True)
model.eta = Param(initialize=3.5, mutable=True)
model.C_el = Param(model.T, initialize=lambda model, t: 0.3, mutable=True)
model.C_FW = Param(initialize=0.1, mutable=True)
model.C_PV = Param(model.T, initialize=lambda model, t: -0.05, mutable=True)

model.Q_PV_max = Param(initialize=5.0, mutable=True)
model.Q_sto_max = Param(initialize=10.0, mutable=True)
model.Q_WP_max = Param(initialize=6.0, mutable=True)
model.Q_FW_max = Param(initialize=10.0, mutable=True)
model.E_sto_init = Param(initialize=5.0, mutable=True)

# Entscheidungsvariablen
model.E_PV_sto = Var(model.T, domain=NonNegativeReals)
model.E_PV_WP = Var(model.T, domain=NonNegativeReals)
model.E_PV_Netz = Var(model.T, domain=NonNegativeReals)
model.E_sto_WP = Var(model.T, domain=NonNegativeReals)
model.E_Netz_WP = Var(model.T, domain=NonNegativeReals)
model.E_sto = Var(model.T, domain=NonNegativeReals)
model.W_FW = Var(model.T, domain=NonNegativeReals)
model.W_WP = Var(model.T, domain=NonNegativeReals)

# Zielfunktion
def objective_rule(model):
    return sum(
        model.C_el[t] * model.E_Netz_WP[t] +
        model.C_FW * model.W_FW[t] +
        model.C_PV[t] * model.E_PV_Netz[t]
        for t in model.T
    )
model.Obj = Objective(rule=objective_rule, sense=minimize)

# Nebenbedingungen
def heat_demand_rule(model, t):
    return model.W_FW[t] + model.W_WP[t] >= model.d[t]
model.heat_demand = Constraint(model.T, rule=heat_demand_rule)

def wp_output_rule(model, t):
    return model.W_WP[t] == model.eta * (
        model.E_PV_WP[t] + model.E_sto_WP[t] + model.E_Netz_WP[t]
    )
model.wp_output = Constraint(model.T, rule=wp_output_rule)

def wp_capacity_rule(model, t):
    return model.W_WP[t] <= model.Q_WP_max
model.wp_capacity = Constraint(model.T, rule=wp_capacity_rule)

def fw_capacity_rule(model, t):
    return model.W_FW[t] <= model.Q_FW_max
model.fw_capacity = Constraint(model.T, rule=fw_capacity_rule)

def pv_distribution_rule(model, t):
    return (
        model.E_PV_WP[t] + model.E_PV_sto[t] + model.E_PV_Netz[t]
        <= model.Q_PV_max
    )
model.pv_distribution = Constraint(model.T, rule=pv_distribution_rule)

def storage_balance_rule(model, t):
    if t == 0:
        return model.E_sto[t] == model.E_sto_init + model.E_PV_sto[t] - model.E_sto_WP[t]
    return model.E_sto[t] == model.E_sto[t - 1] + model.E_PV_sto[t] - model.E_sto_WP[t]
model.storage_balance = Constraint(model.T, rule=storage_balance_rule)

def storage_capacity_rule(model, t):
    return model.E_sto[t] <= model.Q_sto_max
model.storage_capacity = Constraint(model.T, rule=storage_capacity_rule)
