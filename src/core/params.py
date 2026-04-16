"""
SimulationParams: physical and LBM parameters derived from Re, u_ref, L_ref.
"""

from typing import NamedTuple

from src.core.lattice import Lattice

class SimulationParams(NamedTuple):
    """
    All parameters needed to define an LBM simulation.

    Attributes
    ----------
    Re     : Reynolds number
    u_ref  : reference (inlet) velocity in LBM units
    L_ref  : reference length in grid cells
    nu     : kinematic viscosity  = u_ref * L_ref / Re
    tau    : relaxation time      = nu / cs² + 0.5
    omega  : relaxation frequency = 1 / tau
    """

    Re:    float
    u_ref: float
    L_ref: float
    nu:    float
    tau:   float
    omega: float

    @staticmethod
    def from_Re(
        Re:      float,
        u_ref:   float,
        L_ref:   float,
        lattice: Lattice,
    ) -> "SimulationParams":
        """
        Construct params from the three independent inputs.

        Parameters
        ----------
        Re     : target Reynolds number
        u_ref  : reference velocity in LBM units (typically 0.05–0.1 for stability)
        L_ref  : characteristic length in grid cells
        lattice: Lattice model used in simulation
        """
        # cs2: lattice speed of sound squared (default 1/3 for standard lattices).
        # Pass lattice.cs2 to ensure consistency with your lattice choice.
        # Determines tau via  nu = cs² * (tau - 0.5)  →  tau = nu/cs² + 0.5
        nu    = u_ref * L_ref / Re
        tau   = nu / lattice.cs2 + 0.5
        omega = 1.0 / tau
        if tau <= 0.5:
            raise ValueError(
                f"tau={tau:.4f} <= 0.5 is unstable. "
                "Reduce u_ref or increase L_ref / Re."
            )
        return SimulationParams(
            Re=Re, u_ref=u_ref, L_ref=L_ref,
            nu=nu, tau=tau, omega=omega,
        )

    def __repr__(self) -> str:
        return (
            f"SimulationParams("
            f"Re={self.Re}, u_ref={self.u_ref}, L_ref={self.L_ref}, "
            f"nu={self.nu:.5f}, tau={self.tau:.5f}, omega={self.omega:.5f})"
        )
