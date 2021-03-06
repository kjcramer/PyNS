"""
Returns running average of the array or matrix sent as parameter.

If array is sent for averaging, only one argument is needed, array itself.

If matrix is sent for averaging, two arguments are needed, matrix itself, and
the desired direction for averaging.

It is clear that the returning average will have one element less in the
direction in which it was averaged:

Example:

  Array sent:    |-------|-------|-------|-------|-------| => six elements
                 0       1       2       3       4       5

  Returnig array:    o-------o-------o-------o-------o     => five elements
                     0       1       2       3       4

  Element 0 in the returning array will be the average between the values
  in elements 1 and 0 in the sending array.
"""

# PyNS modules
from pyns.constants.coordinates import X, Y, Z

# =============================================================================
def avg(*args):
# -----------------------------------------------------------------------------
    """
    Args: Number of input arguments can be one or two, depending if one wants
          to run an average on array or matrix.

      One argument provided (for arrays)
        args[0]: Array for averaging.

      Two arguments provided (for matrices)
        args[0]: Matrix for averaging.
        args[1]: Direction for averaging (X, Y or Z)

    Returns:
      Array or matrix with averaged values.

    Note:
      Separate functions for averaging in "x", "y" and "z" directions
      are also defined.  They require fewer arguments being passed and
      shorter syntax.  This function, however, is still usefull in
      instances in which the averaging direction is not predefined.
    """

    # Only one argument is sent - perform
    # averaging of one-dimensional array
    if len((args)) == 1:
        a = args[0]
        return (a[1:] + a[:-1]) * 0.5

    # Two arguments are sent - perform averaging of a
    # three-dimensional array, with a given direction
    elif len((args)) == 2:
        dir = args[0]  # direction
        a   = args[1]  # array

        if dir == X:
            return (a[1:,:,:] + a[:-1,:,:]) * 0.5
        elif dir == Y:
            return (a[:,1:,:] + a[:,:-1,:]) * 0.5
        elif dir == Z:
            return (a[:,:,1:] + a[:,:,:-1]) * 0.5

    # Some error message might
    # be printed if number of
    # arguments is wrong

    return a  # end of function
    
# =============================================================================
def avg_x(a):
# -----------------------------------------------------------------------------
    """
    Args:
      a: matrix for averaging.

    Returns:
      Matrix with averaged values.
    """

    return (a[1:,:,:] + a[:-1,:,:]) * 0.5  # end of function

# =============================================================================
def avg_y(a):
# -----------------------------------------------------------------------------
    """
    Args:
      a: matrix for averaging.

    Returns:
      Matrix with averaged values.
    """

    return (a[:,1:,:] + a[:,:-1,:]) * 0.5  # end of function
  
# =============================================================================
def avg_z(a):
# -----------------------------------------------------------------------------
    """
    Args:
      a: matrix for averaging.

    Returns:
      Matrix with averaged values.
    """

    return (a[:,:,1:] + a[:,:,:-1]) * 0.5  # end of function
