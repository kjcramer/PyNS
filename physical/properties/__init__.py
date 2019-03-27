"""
This is ruthless violation of Python's ethical code of conduct (best practice).
It includes all the functionality defined in the module "physical_models".
Very bad, but it makes code development faster for non-IT oriendted minds.
"""

# Operators
from .air   import air
from .water import water
from .latent_heat import latent_heat
from .p_v_sat import p_v_sat, p_v_sat_salt, t_sat, t_sat_salt
from .rho_salt import rho_salt
